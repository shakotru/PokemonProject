from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)


# 1. Configuration
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "pokemon.csv"
OUTPUT_DIR = PROJECT_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = [
    "hp", "attack", "defense",
    "sp_attack", "sp_defense", "speed",
]
TYPE_FEATURES = ["type1", "type2"]
FEATURES = NUMERIC_FEATURES + TYPE_FEATURES
TARGET = "is_legendary"
RANDOM_STATE = 101


# 2. Load and inspect the dataset
df = pd.read_csv(DATA_FILE)

required_columns = FEATURES + [TARGET]
missing_columns = sorted(set(required_columns) - set(df.columns))

if missing_columns:
    raise ValueError(
        f"Missing expected columns: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )

print("\nDATASET SHAPE:", df.shape)
print("\nTARGET COUNTS:")
print(df[TARGET].value_counts(dropna=False))
print("\nMISSING VALUES IN SELECTED COLUMNS:")
print(df[required_columns].isna().sum())

duplicate_count = df.duplicated().sum()
print("\nEXACT DUPLICATE ROWS:", duplicate_count)
df = df.drop_duplicates().copy()

# Stop rather than silently guessing how to handle problematic labels.
if df[TARGET].isna().any():
    raise ValueError("The target contains missing labels. Inspect them first.")

if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError("Expected is_legendary to contain only 0 and 1.")

df[TARGET] = df[TARGET].astype(int)

for column in NUMERIC_FEATURES:
    df[column] = pd.to_numeric(df[column], errors="raise")

# Check whether multiple records share the same species identifier.
if "pokedex_number" in df.columns:
    if df["pokedex_number"].dropna().duplicated().any():
        raise ValueError(
            "Repeated Pokédex numbers found. Inspect alternate forms "
            "before using a random row split."
        )


# 3. Reserve 20% as the final test set
development_df, test_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df[TARGET],
    random_state=RANDOM_STATE,
)

# Split the remaining 80% into training and validation.
# Overall proportions: 60% train, 20% validation, 20% test.
train_df, validation_df = train_test_split(
    development_df,
    test_size=0.25,
    stratify=development_df[TARGET],
    random_state=RANDOM_STATE,
)

split_labels = pd.Series(index=df.index, dtype="object")
split_labels.loc[train_df.index] = "train"
split_labels.loc[validation_df.index] = "validation"
split_labels.loc[test_df.index] = "test"

split_manifest = pd.DataFrame({
    "source_row_index": df.index,
    "split": split_labels.loc[df.index].to_numpy(),
})

if "name" in df.columns:
    split_manifest["name"] = df["name"].to_numpy()

split_manifest.to_csv(OUTPUT_DIR / "split_manifest.csv", index=False)

print(
    f"\nSPLIT SIZES: train={len(train_df)}, "
    f"validation={len(validation_df)}, test={len(test_df)}"
)


# 4. Prepare training and validation inputs
X_train = train_df[FEATURES].copy()
X_validation = validation_df[FEATURES].copy()

y_train = train_df[TARGET]
y_validation = validation_df[TARGET]

# Learn numeric replacement values from training records only.
training_medians = X_train[NUMERIC_FEATURES].median()

if training_medians.isna().any():
    raise ValueError("A numeric feature is entirely missing in training data.")

X_train[NUMERIC_FEATURES] = (
    X_train[NUMERIC_FEATURES].fillna(training_medians)
)
X_validation[NUMERIC_FEATURES] = (
    X_validation[NUMERIC_FEATURES].fillna(training_medians)
)

# Missing secondary type is treated as no secondary type.
# A missing primary type is treated as unknown.
for frame in [X_train, X_validation]:
    frame["type1"] = frame["type1"].fillna("Unknown")
    frame["type2"] = frame["type2"].fillna("None")

# Learn the encoded column layout from training data.
X_train = pd.get_dummies(
    X_train, columns=TYPE_FEATURES, dtype=float
)
X_validation = pd.get_dummies(
    X_validation, columns=TYPE_FEATURES, dtype=float
)

X_validation = X_validation.reindex(
    columns=X_train.columns, fill_value=0
)

# Same scaling sequence as the lecture notebook:
# fit on training data, transform validation data.
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_validation = scaler.transform(X_validation)


# 5. Fit the reference and learned baseline
models = {
    "Majority reference": DummyClassifier(strategy="most_frequent"),
    "Logistic regression": LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    ),
}

metric_rows = []

for model_name, model in models.items():
    model.fit(scaled_X_train, y_train)

    predictions = model.predict(scaled_X_validation)
    positive_column = list(model.classes_).index(1)
    probabilities = model.predict_proba(
        scaled_X_validation
    )[:, positive_column]

    metric_rows.append({
        "model": model_name,
        "accuracy": accuracy_score(y_validation, predictions),
        "precision": precision_score(
            y_validation, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_validation, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_validation, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_validation, probabilities),
    })

metrics = pd.DataFrame(metric_rows)
print("\nVALIDATION RESULTS:")
print(metrics.round(4).to_string(index=False))
metrics.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)


# 6. Detailed logistic-regression evaluation
log_model = models["Logistic regression"]
y_pred = log_model.predict(scaled_X_validation)
positive_column = list(log_model.classes_).index(1)
y_probability = log_model.predict_proba(
    scaled_X_validation
)[:, positive_column]

report = classification_report(
    y_validation,
    y_pred,
    labels=[0, 1],
    target_names=["Non-legendary", "Legendary"],
    zero_division=0,
)

print("\nLOGISTIC REGRESSION CLASSIFICATION REPORT:")
print(report)
(OUTPUT_DIR / "classification_report.txt").write_text(
    report, encoding="utf-8"
)


# 7. Save evaluation plots
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ConfusionMatrixDisplay.from_predictions(
    y_validation,
    y_pred,
    labels=[0, 1],
    display_labels=["Non-legendary", "Legendary"],
    cmap="Blues",
    colorbar=False,
    ax=axes[0],
)
axes[0].set_title("Validation confusion matrix")

RocCurveDisplay.from_predictions(
    y_validation, y_probability, ax=axes[1]
)
axes[1].plot([0, 1], [0, 1], "k--", label="Chance")
axes[1].legend()
axes[1].set_title("Validation ROC curve")

PrecisionRecallDisplay.from_predictions(
    y_validation, y_probability, ax=axes[2]
)
axes[2].set_title("Validation precision–recall curve")

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "validation_plots.png", dpi=200)
plt.close(fig)


# 8. Save individual predictions and coefficients
prediction_table = pd.DataFrame(
    index=validation_df.index
)

if "name" in validation_df.columns:
    prediction_table["name"] = validation_df["name"]

prediction_table["actual_legendary"] = y_validation
prediction_table["predicted_legendary"] = y_pred
prediction_table["legendary_probability"] = y_probability
prediction_table["correct"] = (
    prediction_table["actual_legendary"]
    == prediction_table["predicted_legendary"]
)

prediction_table.to_csv(
    OUTPUT_DIR / "validation_predictions.csv",
    index_label="source_row_index",
)

coefficient_table = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": log_model.coef_[0],
})
coefficient_table["absolute_coefficient"] = (
    coefficient_table["coefficient"].abs()
)
coefficient_table = coefficient_table.sort_values(
    "absolute_coefficient", ascending=False
)
coefficient_table.to_csv(
    OUTPUT_DIR / "coefficients.csv", index=False
)

print("\nMISCLASSIFIED VALIDATION POKEMON:")
errors = prediction_table.loc[~prediction_table["correct"]]
print(errors.to_string())

print(f"\nResults saved in: {OUTPUT_DIR}")
print("The final test set has NOT been evaluated.")