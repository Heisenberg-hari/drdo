import pandas as pd
import numpy as np
from pathlib import Path
import time
import matplotlib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

FAST_MODE = True
USE_POLYNOMIAL_FEATURES = False

# --------------------------------
# 1. Load Dataset
# --------------------------------
dataset_path = Path(__file__).resolve().parent / "clean_dataset_.xlsx"
data = pd.read_excel(dataset_path)

required_cols = {
    "pulse_bpm", "temperature_c", "external_pressure",
    "body_pressure", "oximeter_reading", "motion_detection", "prediction"
}
missing_required = required_cols.difference(data.columns)
if missing_required:
    raise ValueError(f"Dataset is missing required columns: {sorted(missing_required)}")

# Quick data checks
print("Rows, cols:", data.shape)
print(data.info())
print(data.describe())
print("Missing values by column:\n", data.isna().sum())

# Drop duplicates and rows with impossible values
data = data.drop_duplicates()

# --------------------------------
# 2. Data preprocessing: Impute + outlier fixes
# --------------------------------
# Choose numeric features from source
num_cols = ['pulse_bpm', 'temperature_c', 'external_pressure', 'body_pressure', 'oximeter_reading', 'motion_detection']

imputer = SimpleImputer(strategy='median')
data[num_cols] = imputer.fit_transform(data[num_cols])

# Clip physically impossible values (domain knowledge)
data['pulse_bpm'] = data['pulse_bpm'].clip(20, 220)
data['temperature_c'] = data['temperature_c'].clip(30.0, 45.0)
data['oximeter_reading'] = data['oximeter_reading'].clip(30, 100)

# --------------------------------
# 3. Feature engineering
# --------------------------------
# ratios and interactions
data['pulse_temp_ratio'] = data['pulse_bpm'] / (data['temperature_c'] + 1e-6)

data['pressure_diff'] = data['external_pressure'] - data['body_pressure']

data['oximeter_drop'] = 100 - data['oximeter_reading']

# flags
data['hypoxia_flag'] = (data['oximeter_reading'] < 90).astype(int)
data['extreme_temp'] = ((data['temperature_c'] < 34) | (data['temperature_c'] > 38)).astype(int)

data['high_motion'] = (data['motion_detection'] > data['motion_detection'].median()).astype(int)

# polynomial terms for non-linear relationships (optional to control overfitting)
if USE_POLYNOMIAL_FEATURES:
    for col in ['pulse_bpm', 'oximeter_reading', 'body_pressure', 'temperature_c']:
        data[f'{col}_sq'] = data[col] ** 2

# target variable
prediction_series = data["prediction"]
if np.issubdtype(prediction_series.dtype, np.number):
    unique_count = prediction_series.nunique(dropna=True)
    if unique_count > 10:
        print(
            f"WARNING: 'prediction' has {unique_count} unique numeric values."
            " Binarizing at 0.5 may produce weak/unstable labels."
        )
    y = (prediction_series > 0.5).astype(int)
else:
    text_map = {
        "alive": 0, "normal": 0, "safe": 0, "healthy": 0, "0": 0,
        "dead": 1, "critical": 1, "unsafe": 1, "1": 1
    }
    y = prediction_series.astype(str).str.strip().str.lower().map(text_map)
    if y.isna().any():
        bad_labels = sorted(prediction_series[y.isna()].dropna().astype(str).unique().tolist())
        raise ValueError(f"Unsupported text labels in 'prediction': {bad_labels}")
    y = y.astype(int)

if y.nunique() < 2:
    raise ValueError("Target 'prediction' has only one class after preprocessing; cannot train classifier.")

feature_cols = num_cols + [
    'pulse_temp_ratio', 'pressure_diff', 'oximeter_drop', 'hypoxia_flag', 'extreme_temp', 'high_motion'
]
if USE_POLYNOMIAL_FEATURES:
    feature_cols += ['pulse_bpm_sq', 'oximeter_reading_sq', 'body_pressure_sq', 'temperature_c_sq']
X = data[feature_cols]

# Quick learnability signal for dataset quality (safe against zero-variance columns)
target_corr_values = []
y_values = y.to_numpy(dtype=float)
y_std = float(np.std(y_values))
for col in X.columns:
    x_values = X[col].to_numpy(dtype=float)
    x_std = float(np.std(x_values))
    if x_std == 0.0 or y_std == 0.0:
        continue
    corr = float(np.corrcoef(x_values, y_values)[0, 1])
    if np.isfinite(corr):
        target_corr_values.append(abs(corr))

max_corr = max(target_corr_values) if target_corr_values else 0.0
print(f"Max absolute feature-target correlation: {max_corr:.4f}")
if max_corr < 0.10:
    print(
        "WARNING: Features have very weak correlation with target. "
        "Dataset labels may be noisy/random, so model accuracy may stay near chance."
    )

# --------------------------------
# 4. Train/validation/test split and fast model fine-tuning
# --------------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

base_model = RandomForestClassifier(
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=1
)

# Smaller randomized search space to avoid long "stuck" runtime
param_dist = {
    "n_estimators": [180, 250, 320],
    "max_depth": [4, 5, 6],
    "min_samples_split": [20, 30, 40, 50],
    "min_samples_leaf": [8, 12, 16, 20],
    "max_features": ["log2", 0.4, 0.5],
    "bootstrap": [True],
    "max_samples": [0.6, 0.7, 0.8],
    "ccp_alpha": [0.0005, 0.001, 0.002]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

if FAST_MODE:
    print("FAST_MODE=True: skipping hyperparameter search and using anti-overfitting parameters.")
    best_model = RandomForestClassifier(
        n_estimators=220,
        max_depth=5,
        min_samples_split=40,
        min_samples_leaf=15,
        max_features=0.5,
        bootstrap=True,
        max_samples=0.7,
        ccp_alpha=0.002,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=1
    )
    best_model.fit(X_train, y_train)
else:
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=30,
        scoring="f1",
        cv=cv,
        n_jobs=1,
        verbose=1,
        random_state=42
    )

    print("Starting model tuning (this may take around 1-3 minutes)...")
    t0 = time.time()
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    elapsed = time.time() - t0
    print(f"Tuning completed in {elapsed:.1f} seconds")

    print("Best CV F1 (dead class):", round(search.best_score_, 4))
    print("Best Params:", search.best_params_)

# Optimize threshold on validation split (not test) for dead-class F1
y_val_proba = best_model.predict_proba(X_val)[:, 1]
threshold_candidates = np.arange(0.10, 0.91, 0.01)
best_threshold = 0.50
best_f1_dead = -1.0
for threshold in threshold_candidates:
    trial_pred = (y_val_proba >= threshold).astype(int)
    trial_f1_dead = f1_score(y_val, trial_pred, pos_label=1, zero_division=0)
    if trial_f1_dead > best_f1_dead:
        best_f1_dead = trial_f1_dead
        best_threshold = float(threshold)

print(f"Optimized probability threshold: {best_threshold:.2f}")
print(f"Validation dead-class F1 at optimized threshold: {best_f1_dead:.4f}")

y_train_proba = best_model.predict_proba(X_train)[:, 1]
y_train_pred = (y_train_proba >= best_threshold).astype(int)
y_val_pred = (y_val_proba >= best_threshold).astype(int)
y_test_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_test_proba >= best_threshold).astype(int)

train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_pred)
train_f1_dead = f1_score(y_train, y_train_pred, pos_label=1, zero_division=0)
val_f1_dead = f1_score(y_val, y_val_pred, pos_label=1, zero_division=0)
test_f1_dead = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

print(f"Train Accuracy: {train_acc:.4f} | Dead F1: {train_f1_dead:.4f}")
print(f"Val Accuracy:   {val_acc:.4f} | Dead F1: {val_f1_dead:.4f}")
print(f"Test Accuracy:  {test_acc:.4f} | Dead F1: {test_f1_dead:.4f}")
print(f"Overfit gap (Train F1 - Test F1): {train_f1_dead - test_f1_dead:.4f}")

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot feature importances for random forest
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    features = X.columns
    fi_df = pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=fi_df, errorbar=None)
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()
    out_plot = Path(__file__).resolve().parent / "feature_importance.png"
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"Saved feature importance plot to: {out_plot}")

# --------------------------------
# 5. Remaining Life Estimation (existing logic)
# --------------------------------
def estimate_remaining_minutes(pulse, spo2, temp, motion, body_pressure):
    if pulse <= 0 or spo2 < 50:
        return 0

    score = (
        0.35 * (pulse / 72)
        + 0.35 * (spo2 / 98)
        + 0.15 * (temp / 36.8)
        + 0.10 * (motion)
        + 0.05 * (body_pressure / 85)
    )

    remaining_minutes = score * 60
    return max(0, remaining_minutes)


# --------------------------------
# 6. User input prediction using tuned model + optimized threshold
# --------------------------------
def read_float(prompt_text):
    while True:
        try:
            return float(input(prompt_text).strip())
        except ValueError:
            print("Invalid input. Please enter a numeric value.")


print("\nEnter sensor values for prediction:")
user_pulse = read_float("pulse_bpm: ")
user_temp = read_float("temperature_c: ")
user_external_pressure = read_float("external_pressure: ")
user_body_pressure = read_float("body_pressure: ")
user_spo2 = read_float("oximeter_reading: ")
user_motion = read_float("motion_detection: ")

new_sample = pd.DataFrame(
    [[user_pulse, user_temp, user_external_pressure, user_body_pressure, user_spo2, user_motion]],
    columns=['pulse_bpm', 'temperature_c', 'external_pressure', 'body_pressure', 'oximeter_reading', 'motion_detection']
)

new_sample['pulse_temp_ratio'] = new_sample['pulse_bpm'] / (new_sample['temperature_c'] + 1e-6)
new_sample['pressure_diff'] = new_sample['external_pressure'] - new_sample['body_pressure']
new_sample['oximeter_drop'] = 100 - new_sample['oximeter_reading']
new_sample['hypoxia_flag'] = (new_sample['oximeter_reading'] < 90).astype(int)
new_sample['extreme_temp'] = ((new_sample['temperature_c'] < 34) | (new_sample['temperature_c'] > 38)).astype(int)
new_sample['high_motion'] = (new_sample['motion_detection'] > data['motion_detection'].median()).astype(int)
for col in ['pulse_bpm', 'oximeter_reading', 'body_pressure', 'temperature_c']:
    if USE_POLYNOMIAL_FEATURES:
        new_sample[f'{col}_sq'] = new_sample[col] ** 2

sample_proba_dead = best_model.predict_proba(new_sample[feature_cols])[:, 1][0]
sample_pred = int(sample_proba_dead >= best_threshold)
print(f"Dead probability: {sample_proba_dead:.4f} (threshold={best_threshold:.2f})")

if sample_pred == 0:
    minutes = estimate_remaining_minutes(user_pulse, user_spo2, user_temp, user_motion, user_body_pressure)
    print("Status: Alive")
    print("Estimated Remaining Minutes:", round(minutes, 2))
else:
    print("Status: Dead")
