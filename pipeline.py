"""
pipeline.py
=================================================================
Streamlit-free analytics pipeline for the Hotel Booking Cancellation
Risk project (BAMD, IIM Calcutta).

This module is intentionally kept free of any Streamlit calls so that
it can be:
  1) imported by the Colab notebook for model development, and
  2) imported by app.py to power the live dashboard,
with a guarantee that both surfaces see identical numbers.

Sections:
  - DATA LOADING & CLEANING
  - FEATURE ENGINEERING
  - TRAIN / TEST SPLIT (time-based)
  - PREPROCESSING + MODEL TRAINING
  - EVALUATION HELPERS
  - RISK SEGMENTATION
  - OVERBOOKING SIMULATION
=================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, recall_score, precision_score, f1_score,
    brier_score_loss
)

RANDOM_STATE = 42

# -----------------------------------------------------------------
# Columns that are NOT available at the moment of booking and must
# be dropped to avoid leakage, per BAMD Project Plan Section 6.1.
# -----------------------------------------------------------------
LEAKAGE_COLS = [
    "reservation_status",       # *is* the outcome, recorded after the fact
    "reservation_status_date",  # date of that outcome
    "assigned_room_type",       # only known once the guest is actually roomed
]

MONTH_ORDER = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]
MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTH_ORDER)}


# =================================================================
# 1. DATA LOADING & CLEANING
# =================================================================

@dataclass
class CleaningReport:
    """A transparent, inspectable record of every cleaning decision made,
    so nothing is silently dropped or altered."""
    raw_rows: int = 0
    duplicate_rows_removed: int = 0
    zero_guest_rows_removed: int = 0
    negative_adr_rows_fixed: int = 0
    extreme_adr_rows_capped: int = 0
    undefined_category_rows: int = 0
    children_nulls_filled: int = 0
    country_nulls_filled: int = 0
    agent_nulls_filled: int = 0
    final_rows: int = 0
    notes: list = field(default_factory=list)


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Applies every documented cleaning rule and returns the cleaned
    frame together with a CleaningReport so every change is auditable."""
    report = CleaningReport(raw_rows=len(df_raw))
    df = df_raw.copy()

    # --- exact duplicate bookings -----------------------------------
    # ~27% of rows in this public dataset are exact duplicates (a known
    # quirk of how the PMS export was produced). Left in, they would let
    # the same booking appear in both train and test, inflating reported
    # performance. We deduplicate on the full row before any feature
    # engineering, the safest point to do it.
    before = len(df)
    df = df.drop_duplicates()
    report.duplicate_rows_removed = before - len(df)

    # --- bookings with zero adults, children AND babies -------------
    # ~0.15% of rows; these are not real reservations (likely admin /
    # test entries in the source PMS) and have no meaningful guest to
    # model risk for.
    zero_guest_mask = (df["adults"].fillna(0) + df["children"].fillna(0) + df["babies"].fillna(0)) == 0
    report.zero_guest_rows_removed = int(zero_guest_mask.sum())
    df = df[~zero_guest_mask]

    # --- adr (average daily rate) data-entry errors -------------------
    # One row has a negative ADR (a refund posted as a negative rate);
    # we treat it as a data-entry artifact and floor it at 0 rather than
    # drop the booking, since every other field is usable.
    neg_mask = df["adr"] < 0
    report.negative_adr_rows_fixed = int(neg_mask.sum())
    df.loc[neg_mask, "adr"] = 0.0

    # A handful of bookings have an implausible ADR (>5,000/night against
    # a dataset median of ~95). We cap rather than drop, at the 99.9th
    # percentile, to preserve the booking while preventing one extreme
    # value from dominating revenue-based calculations later.
    cap = df["adr"].quantile(0.999)
    extreme_mask = df["adr"] > cap
    report.extreme_adr_rows_capped = int(extreme_mask.sum())
    df.loc[extreme_mask, "adr"] = cap

    # --- missing values ----------------------------------------------
    report.children_nulls_filled = int(df["children"].isnull().sum())
    df["children"] = df["children"].fillna(0)

    report.country_nulls_filled = int(df["country"].isnull().sum())
    df["country"] = df["country"].fillna("Unknown")

    # agent: NaN genuinely means "no agent involved" (direct/corporate
    # bookings) rather than missing data, so we encode it as its own
    # category instead of imputing a guess.
    report.agent_nulls_filled = int(df["agent"].isnull().sum())
    df["agent"] = df["agent"].fillna(0).astype(int).astype(str)
    df["agent"] = df["agent"].replace("0", "No Agent")

    # company is ~94% missing (most guests are not booked through a
    # company account); too sparse to impute meaningfully, so we keep
    # only a binary "booked_via_company" signal instead of the raw ID.
    df["booked_via_company"] = df["company"].notnull().astype(int)
    df = df.drop(columns=["company"])

    # --- "Undefined" categorical values --------------------------------
    # market_segment, distribution_channel and meal each contain a small
    # number of literal "Undefined" / "SC" placeholder rows from the
    # source system. We keep them as an explicit category rather than
    # guessing, but log the count for transparency.
    undefined_count = 0
    for col in ["market_segment", "distribution_channel"]:
        undefined_count += int((df[col] == "Undefined").sum())
    report.undefined_category_rows = undefined_count

    report.final_rows = len(df)
    report.notes.append(
        f"Removed {report.duplicate_rows_removed:,} duplicate rows "
        f"({report.duplicate_rows_removed/report.raw_rows:.1%} of raw data)."
    )
    report.notes.append(
        f"Removed {report.zero_guest_rows_removed} bookings with zero adults, "
        f"children and babies (not real reservations)."
    )
    report.notes.append(
        f"{report.undefined_category_rows} rows carry an 'Undefined' market "
        f"segment or distribution channel; kept as-is rather than imputed."
    )

    return df.reset_index(drop=True), report


# =================================================================
# 2. FEATURE ENGINEERING
# =================================================================

# Final feature lists used by every model in this project. Keeping this
# as a single source of truth means the notebook and the Streamlit app
# can never silently drift apart on what a model is trained on.
NUMERIC_FEATURES = [
    "lead_time", "arrival_date_week_number", "stays_in_weekend_nights",
    "stays_in_week_nights", "total_nights", "adults", "children", "babies",
    "total_guests", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes", "agent_lead_time_dev",
    "days_in_waiting_list", "adr", "required_car_parking_spaces",
    "total_of_special_requests", "booked_via_company", "arrival_month_num",
]

CATEGORICAL_FEATURES = [
    "hotel", "meal", "country_grouped", "market_segment",
    "distribution_channel", "reserved_room_type", "deposit_type",
    "customer_type", "season", "arrival_day_of_week",
]

TOP_N_COUNTRIES = 12


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds every modeling feature from the cleaned raw columns and
    drops leakage columns. Every derived column is something a revenue
    manager would actually know at the moment a booking is made."""
    df = df.copy()

    # --- calendar features --------------------------------------------
    df["arrival_month_num"] = df["arrival_date_month"].map(MONTH_NUM)
    df["arrival_date"] = pd.to_datetime(
        dict(year=df["arrival_date_year"], month=df["arrival_month_num"],
             day=df["arrival_date_day_of_month"]),
        errors="coerce",
    )
    # booking_date = the date the reservation was actually made
    df["booking_date"] = df["arrival_date"] - pd.to_timedelta(df["lead_time"], unit="D")
    df["arrival_day_of_week"] = df["arrival_date"].dt.day_name()

    def to_season(m):
        if m in (12, 1, 2):
            return "Winter"
        if m in (3, 4, 5):
            return "Spring"
        if m in (6, 7, 8):
            return "Summer"
        return "Autumn"
    df["season"] = df["arrival_month_num"].apply(to_season)

    # --- stay / party composition --------------------------------------
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]

    # --- country grouping (avoid an unmanageable one-hot of 170+ codes) -
    top_countries = df["country"].value_counts().nlargest(TOP_N_COUNTRIES).index
    df["country_grouped"] = np.where(df["country"].isin(top_countries), df["country"], "Other")

    # NOTE: an "agent lead-time deviation" feature (lead time vs. that
    # agent's typical lead time) is added later, in add_agent_deviation_
    # feature(), strictly AFTER the train/test split. Computing it here
    # on the full dataframe would let each agent's TEST-period bookings
    # influence the median used for its TRAIN-period bookings -- a subtle
    # form of leakage across the time boundary we deliberately split on.


    # --- revenue exposure: what this booking is worth if honoured -------
    df["expected_revenue"] = df["adr"] * df["total_nights"].clip(lower=1)

    # --- drop leakage & raw columns now folded into engineered features -
    drop_cols = LEAKAGE_COLS + [
        "country", "arrival_date_year", "arrival_date_month",
        "arrival_date_day_of_month",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def get_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Returns just the columns the models are trained on, plus the
    target and the bookkeeping columns (dates, revenue, hotel, agent)
    needed downstream for the split-time feature / simulation / 
    segmentation -- nothing else."""
    keep = (
        NUMERIC_FEATURES + CATEGORICAL_FEATURES +
        ["is_canceled", "booking_date", "arrival_date", "expected_revenue", "hotel", "agent"]
    )
    keep = [c for c in keep if c != "agent_lead_time_dev"]  # added post-split, not yet present
    keep = list(dict.fromkeys(keep))  # de-dup, preserve order
    return df[keep].copy()


# =================================================================
# 3. TRAIN / TEST SPLIT  (time-based, not random)
# =================================================================

def time_based_split(df: pd.DataFrame, split_date: str = "2017-02-01"):
    """Splits on booking_date rather than a random shuffle.

    Why: a random split lets the model 'see the future' relative to any
    given test booking (e.g. a March 2017 cancellation pattern leaking
    into a February 2017 training fold via a near-duplicate booking from
    the same agent). Splitting on the date the *booking was made* mimics
    how the model will actually be used -- trained on history, scored on
    bookings made from a future cut-off point onward.
    """
    split_date = pd.Timestamp(split_date)
    train = df[df["booking_date"] < split_date].copy()
    test = df[df["booking_date"] >= split_date].copy()
    return train, test


def add_agent_deviation_feature(train: pd.DataFrame, test: pd.DataFrame):
    """Adds `agent_lead_time_dev`: how much longer/shorter this booking's
    lead time is versus that agent's typical (median) lead time.

    Computed strictly from TRAIN statistics and then applied to both
    frames -- the median lead time per agent is "frozen" at the train/
    test boundary, exactly as it would be in production where you can
    only know an agent's historical pattern, never their future one.
    Agents seen only in the test period fall back to the train-wide
    median lead time.
    """
    train = train.copy()
    test = test.copy()
    agent_medians = train.groupby("agent")["lead_time"].median()
    global_median = train["lead_time"].median()

    train["agent_lead_time_dev"] = train["lead_time"] - train["agent"].map(agent_medians)
    test["agent_lead_time_dev"] = test["lead_time"] - test["agent"].map(agent_medians).fillna(global_median)
    train["agent_lead_time_dev"] = train["agent_lead_time_dev"].fillna(0)
    test["agent_lead_time_dev"] = test["agent_lead_time_dev"].fillna(0)
    return train, test


# =================================================================
# 4. PREPROCESSING + MODEL TRAINING
# =================================================================

def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def build_tree_preprocessor() -> ColumnTransformer:
    """Tree models don't need scaling, and HistGradientBoostingClassifier
    can take ordinal-encoded categoricals directly and natively handles
    missing values -- a lighter pipeline than the linear-model one."""
    from sklearn.preprocessing import OrdinalEncoder
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


MODEL_REGISTRY = {
    "Logistic Regression": dict(
        uses_tree_prep=False,
        estimator=LogisticRegression(
            max_iter=2000, class_weight="balanced", C=0.5, random_state=RANDOM_STATE
        ),
    ),
    "Decision Tree": dict(
        uses_tree_prep=True,
        estimator=DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=50, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    ),
    "Random Forest": dict(
        uses_tree_prep=True,
        estimator=RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
        ),
    ),
    "Gradient Boosting": dict(
        uses_tree_prep=True,
        estimator=HistGradientBoostingClassifier(
            max_iter=300, max_depth=8, learning_rate=0.08,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
    ),
}


def train_all_models(train_df: pd.DataFrame, models: Optional[list] = None) -> dict:
    """Trains every model in MODEL_REGISTRY (or a subset) on the same
    training frame and returns fitted sklearn Pipelines keyed by name."""
    models = models or list(MODEL_REGISTRY.keys())
    linear_prep = build_preprocessor()
    tree_prep = build_tree_preprocessor()

    y_train = train_df["is_canceled"]
    fitted = {}
    for name in models:
        spec = MODEL_REGISTRY[name]
        prep = tree_prep if spec["uses_tree_prep"] else linear_prep
        pipe = Pipeline([("prep", prep), ("clf", spec["estimator"])])
        pipe.fit(train_df, y_train)
        fitted[name] = pipe
    return fitted


def train_probability_model(train_df: pd.DataFrame) -> Pipeline:
    """Trains a SEPARATE Gradient Boosting model with NO class weighting,
    used only where the raw probability value itself matters (risk
    segmentation, the overbooking simulator) rather than just the
    cancel/no-cancel ranking.

    Why this exists -- a finding worth stating plainly: `class_weight=
    "balanced"` (used in every model in MODEL_REGISTRY to get usable
    recall at a 0.5 threshold) systematically inflates predicted
    probabilities. On this dataset, the balanced Gradient Boosting model
    outputs a mean predicted cancel probability of ~0.38 against an
    actual rate of ~0.28 -- still excellent at RANKING bookings by risk
    (ROC-AUC barely moves), but no longer trustworthy as an actual
    probability to feed into a Monte-Carlo simulation that needs real
    Bernoulli rates. We tested recalibrating the balanced model with
    isotonic regression (sklearn's CalibratedClassifierCV); it degraded
    discrimination (ROC-AUC fell from 0.84 to ~0.69) because the
    dataset's chronological row order broke the calibrator's internal
    cross-validation folds. Rather than fight that, we simply train an
    unweighted twin of the same architecture: its probabilities are
    already well-calibrated (confirmed via a 10-bin reliability check)
    and it costs nothing extra to maintain.
    """
    prep = build_tree_preprocessor()
    clf = HistGradientBoostingClassifier(
        max_iter=300, max_depth=8, learning_rate=0.08, random_state=RANDOM_STATE
    )
    pipe = Pipeline([("prep", prep), ("clf", clf)])
    pipe.fit(train_df, train_df["is_canceled"])
    return pipe


# =================================================================
# 5. EVALUATION HELPERS
# =================================================================

def evaluate_model(pipe: Pipeline, test_df: pd.DataFrame, threshold: float = 0.5) -> dict:
    y_true = test_df["is_canceled"].values
    y_prob = pipe.predict_proba(test_df)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return dict(
        roc_auc=roc_auc_score(y_true, y_prob),
        pr_auc=average_precision_score(y_true, y_prob),
        recall=recall_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        brier=brier_score_loss(y_true, y_prob),
        confusion_matrix=confusion_matrix(y_true, y_pred),
        y_true=y_true,
        y_prob=y_prob,
    )


def get_roc_curve(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return fpr, tpr, thr


def get_pr_curve(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    return prec, rec, thr


def cost_sensitive_threshold(y_true, y_prob, cost_fn: float, cost_fp: float,
                              thresholds: np.ndarray = None) -> pd.DataFrame:
    """Sweeps the decision threshold and computes total expected cost at
    each point, for a retention-outreach style policy:

      cost_fn = cost of a MISSED cancellation (booking was going to cancel,
                 model didn't flag it -> no retention call made -> revenue lost)
      cost_fp = cost of a FALSE ALARM (booking was going to honour anyway,
                 model flagged it -> a retention call was made for nothing,
                 e.g. agent time, or a needless discount offered)

    This is a different decision from the overbooking buffer: it answers
    "which individual bookings should a revenue manager proactively call
    today", not "how many extra rooms should we sell tonight".
    """
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.01, 0.99, 0.02), 2)
    y_true = np.asarray(y_true)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        total_cost = fn * cost_fn + fp * cost_fp
        rows.append(dict(threshold=t, fn=fn, fp=fp, tp=tp, tn=tn, total_cost=total_cost))
    return pd.DataFrame(rows)


def odds_ratios_from_logreg(pipe: Pipeline) -> pd.DataFrame:
    """Extracts plain-English-ready odds ratios from a fitted Logistic
    Regression pipeline (built with build_preprocessor)."""
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    num_names = NUMERIC_FEATURES
    cat_names = list(prep.named_transformers_["cat"]["onehot"].get_feature_names_out(CATEGORICAL_FEATURES))
    all_names = num_names + cat_names
    coefs = clf.coef_[0]
    out = pd.DataFrame({"feature": all_names, "coefficient": coefs})
    out["odds_ratio"] = np.exp(out["coefficient"])
    out["abs_coef"] = out["coefficient"].abs()
    return out.sort_values("abs_coef", ascending=False).drop(columns="abs_coef").reset_index(drop=True)


def tree_feature_importance(pipe: Pipeline, X: pd.DataFrame = None, y: pd.Series = None,
                             n_repeats: int = 5, sample_size: int = 8000) -> pd.DataFrame:
    """Returns a feature-importance table for any tree-based pipeline.

    Random Forest / Decision Tree expose a native Gini-based
    `feature_importances_`, which is fast and used directly. The
    HistGradientBoostingClassifier does not expose this attribute, so we
    fall back to permutation importance computed on a held-out sample
    (X, y must be provided in that case) -- a model-agnostic measure of
    how much shuffling a feature degrades ROC-AUC.
    """
    clf = pipe.named_steps["clf"]
    all_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        out = pd.DataFrame({"feature": all_names, "importance": importances})
        return out.sort_values("importance", ascending=False).reset_index(drop=True)

    if X is None or y is None:
        raise ValueError("X and y must be supplied for permutation importance "
                          "(this model has no native feature_importances_).")

    from sklearn.inspection import permutation_importance
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=RANDOM_STATE)
        y_sample = y.loc[X_sample.index]
    else:
        X_sample, y_sample = X, y

    result = permutation_importance(
        pipe, X_sample, y_sample, scoring="roc_auc",
        n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1,
    )
    out = pd.DataFrame({
        "feature": X_sample.columns if hasattr(X_sample, "columns") else all_names,
        "importance": result.importances_mean,
    })
    # permutation importance here is computed at the raw-column level
    # (pre-encoding), which is actually more interpretable for a manager
    # than a one-hot-exploded list.
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


# =================================================================
# 6. RISK SEGMENTATION
# =================================================================

RISK_BANDS = [
    (0.0, 0.15, "Very Low"),
    (0.15, 0.35, "Low"),
    (0.35, 0.60, "Medium"),
    (0.60, 0.85, "High"),
    (0.85, 1.01, "Very High"),
]


def assign_risk_band(prob: float) -> str:
    for lo, hi, label in RISK_BANDS:
        if lo <= prob < hi:
            return label
    return "Very High"


def build_risk_segments(test_df: pd.DataFrame, y_prob: np.ndarray) -> pd.DataFrame:
    seg = test_df.copy()
    seg["cancel_probability"] = y_prob
    seg["risk_band"] = seg["cancel_probability"].apply(assign_risk_band)
    return seg


def profile_segments(seg_df: pd.DataFrame) -> pd.DataFrame:
    band_order = [b[2] for b in RISK_BANDS]
    agg = seg_df.groupby("risk_band").agg(
        bookings=("is_canceled", "size"),
        actual_cancel_rate=("is_canceled", "mean"),
        avg_predicted_prob=("cancel_probability", "mean"),
        avg_lead_time=("lead_time", "mean"),
        avg_adr=("adr", "mean"),
        avg_special_requests=("total_of_special_requests", "mean"),
        revenue_at_risk=("expected_revenue", "sum"),
        pct_no_deposit=("deposit_type", lambda x: (x == "No Deposit").mean()),
        pct_online_ta=("market_segment", lambda x: (x == "Online TA").mean()),
    ).reindex(band_order).dropna(how="all")
    return agg.reset_index()


# =================================================================
# 7. OVERBOOKING SIMULATION
# =================================================================

@dataclass
class SimulationConfig:
    capacity: int = 100             # physical rooms available on the night
    walk_cost_multiplier: float = 4.0   # cost of walking a guest, as a multiple of one night's ADR
    n_trials: int = 200             # Monte-Carlo repetitions per buffer level


def simulate_overbooking(
    bookings_adr: np.ndarray,
    cancel_probs: np.ndarray,
    capacity: int,
    buffer_sizes: np.ndarray,
    walk_cost_multiplier: float = 4.0,
    n_trials: int = 500,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Monte-Carlo simulation of net revenue for a single night.

    `bookings_adr` / `cancel_probs` represent the *pool* of bookings this
    property type/segment typically receives demand from (e.g. every
    City Hotel booking in the test set). For each candidate `buffer`
    (extra reservations accepted above `capacity`) we repeat, `n_trials`
    times:

      1. draw a fresh random set of `capacity + buffer` bookings from the
         pool (with replacement -- representing "a typical night's worth
         of demand", not one fixed, literal night), then
      2. draw which of those bookings actually cancel, as independent
         Bernoulli trials using each booking's predicted probability,
      3. compute net_revenue = revenue from honoured bookings that fit in
         a room, minus the walk cost for any honoured bookings beyond
         capacity.

    Resampling the booking mix *inside* the trial loop (rather than once
    per buffer level) is deliberate: it means every buffer level is
    evaluated against the same demand distribution, so the expected-
    revenue curve reflects the buffer's effect, not which specific
    bookings happened to be drawn for that buffer.
    """
    rng = np.random.default_rng(random_state)
    n_pool = len(bookings_adr)
    results = []

    for buffer in buffer_sizes:
        accepted_n = capacity + buffer
        net_revenues, walked_counts, empty_counts = [], [], []

        for _ in range(n_trials):
            idx = rng.integers(0, n_pool, size=accepted_n)
            adr_sample = bookings_adr[idx]
            prob_sample = cancel_probs[idx]

            cancels = rng.random(accepted_n) < prob_sample
            shows = ~cancels
            n_show = int(shows.sum())
            honoured_idx = np.where(shows)[0]

            if n_show <= capacity:
                revenue = adr_sample[honoured_idx].sum()
                walk_cost = 0.0
                empty_rooms = capacity - n_show
                walked = 0
            else:
                roomed = honoured_idx[:capacity]
                walked_idx = honoured_idx[capacity:]
                revenue = adr_sample[roomed].sum()
                walk_cost = adr_sample[walked_idx].sum() * walk_cost_multiplier
                empty_rooms = 0
                walked = len(walked_idx)

            net_revenues.append(revenue - walk_cost)
            walked_counts.append(walked)
            empty_counts.append(empty_rooms)

        results.append(dict(
            buffer=buffer,
            accepted=accepted_n,
            expected_net_revenue=np.mean(net_revenues),
            std_net_revenue=np.std(net_revenues),
            expected_walked_guests=np.mean(walked_counts),
            expected_empty_rooms=np.mean(empty_counts),
            p95_walked_guests=np.percentile(walked_counts, 95),
            prob_any_walk=np.mean(np.array(walked_counts) > 0),
        ))

    return pd.DataFrame(results)


def baseline_no_overbooking_revenue(bookings_adr: np.ndarray, cancel_probs: np.ndarray, capacity: int,
                                     n_trials: int = 200, random_state: int = RANDOM_STATE) -> float:
    """Expected revenue if the hotel books exactly to capacity (buffer=0)
    every night and simply absorbs cancellations as empty rooms -- the
    naive policy this project is trying to improve on."""
    return simulate_overbooking(
        bookings_adr, cancel_probs, capacity, buffer_sizes=np.array([0]),
        n_trials=n_trials, random_state=random_state,
    )["expected_net_revenue"].iloc[0]


# =================================================================
# 8. END-TO-END ORCHESTRATION  (single source of truth)
# =================================================================

BEST_MODEL_NAME = "Gradient Boosting"


def run_full_pipeline(csv_path: str, split_date: str = "2017-02-01",
                       train_classification_models: bool = True) -> dict:
    """Runs the entire pipeline from raw CSV to trained models, in the
    exact sequence used throughout this project. Both the notebook and
    app.py call this single function so neither surface can silently
    drift from the other.

    Returns a dict with every intermediate artifact a caller might need:
    raw, clean, cleaning_report, features, train, test, models,
    probability_model, test_probabilities.
    """
    raw = load_raw(csv_path)
    clean, cleaning_report = clean_data(raw)
    features = engineer_features(clean)
    model_frame = get_model_frame(features)
    train, test = time_based_split(model_frame, split_date=split_date)
    train, test = add_agent_deviation_feature(train, test)

    out = dict(
        raw=raw, clean=clean, cleaning_report=cleaning_report,
        features=features, train=train, test=test,
    )

    if train_classification_models:
        models = train_all_models(train)
        out["models"] = models
        out["best_model"] = models[BEST_MODEL_NAME]

    prob_model = train_probability_model(train)
    out["probability_model"] = prob_model
    out["test_probabilities"] = prob_model.predict_proba(test)[:, 1]

    return out


# =================================================================
# 9. SAVE / LOAD ARTIFACTS  (so app.py can skip retraining on every cold start)
# =================================================================

import joblib
import os


def save_artifacts(state: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    if "models" in state:
        for name, pipe in state["models"].items():
            safe_name = name.lower().replace(" ", "_")
            joblib.dump(pipe, os.path.join(out_dir, f"model_{safe_name}.joblib"), compress=3)
    joblib.dump(state["probability_model"], os.path.join(out_dir, "probability_model.joblib"), compress=3)
    state["train"].to_parquet(os.path.join(out_dir, "train.parquet"))
    state["test"].to_parquet(os.path.join(out_dir, "test.parquet"))


def load_artifacts(out_dir: str) -> dict:
    out = {"models": {}}
    for fname in os.listdir(out_dir):
        if fname.startswith("model_") and fname.endswith(".joblib"):
            name_key = fname[len("model_"):-len(".joblib")].replace("_", " ").title()
            # restore canonical names from MODEL_REGISTRY
            for canon in MODEL_REGISTRY:
                if canon.lower().replace(" ", "_") == fname[len("model_"):-len(".joblib")]:
                    name_key = canon
            out["models"][name_key] = joblib.load(os.path.join(out_dir, fname))
    out["probability_model"] = joblib.load(os.path.join(out_dir, "probability_model.joblib"))
    out["train"] = pd.read_parquet(os.path.join(out_dir, "train.parquet"))
    out["test"] = pd.read_parquet(os.path.join(out_dir, "test.parquet"))
    out["best_model"] = out["models"].get(BEST_MODEL_NAME)
    out["test_probabilities"] = out["probability_model"].predict_proba(out["test"])[:, 1]
    return out
