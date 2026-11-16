"""
Stroke risk model: logistic regression (the linear model for a 0/1 target).

Workflow (matches Sections 4-8 of stroke_project_overview.md):
  1. Load the provided train / validation / test splits (random_state=101, stratified).
  2. Baselines: majority class, plain logistic regression, balanced logistic regression.
  3. Tune logistic regression with GridSearchCV (5-fold stratified CV on TRAIN only).
  4. Choose the decision threshold on the VALIDATION set only.
  5. Evaluate once on the TEST set.
  6. Explainability: coefficients / odds ratios, worked example.
  7. Optional check: with and without extreme BMI (> 60).

Run:  py stroke_model.py   (Windows)   or   python3 stroke_model.py
Put the four CSVs in the same folder (or change DATA_DIR).
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

warnings.filterwarnings("ignore")

DATA_DIR = Path(".")
OUT_DIR = Path("model_output")
OUT_DIR.mkdir(exist_ok=True)
SEED = 101

# ----------------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------------
def load(name):
    df = pd.read_csv(DATA_DIR / f"stroke_{name}_scaled.csv")
    rows = df["original_row"].to_numpy()          # identifier only, never a feature
    y = df["stroke"].astype(int).to_numpy()
    X = df.drop(columns=["original_row", "stroke"]).astype(float)
    return X, y, rows

X_tr, y_tr, r_tr = load("train")
X_va, y_va, r_va = load("validation")
X_te, y_te, r_te = load("test")
features = X_tr.columns.tolist()
print(f"train {X_tr.shape}  val {X_va.shape}  test {X_te.shape}")
print(f"strokes: train {y_tr.sum()}  val {y_va.sum()}  test {y_te.sum()}")

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def metrics(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "threshold": thr,
    }

# ----------------------------------------------------------------------------
# 2. Baselines
# ----------------------------------------------------------------------------
rows = {}
dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
p = dummy.predict_proba(X_va)[:, 1]
rows["Majority class"] = {"accuracy": accuracy_score(y_va, dummy.predict(X_va)),
                          "precision": 0.0, "recall": 0.0, "f1": 0.0,
                          "roc_auc": 0.5, "pr_auc": y_va.mean(), "threshold": np.nan}

plain = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
rows["LogReg (plain, thr 0.50)"] = metrics(y_va, plain.predict_proba(X_va)[:, 1])

bal = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_tr, y_tr)
rows["LogReg (balanced, thr 0.50)"] = metrics(y_va, bal.predict_proba(X_va)[:, 1])

# ----------------------------------------------------------------------------
# 3. Tune with GridSearchCV on TRAIN only (5-fold stratified)
#    Scoring = average precision (PR-AUC): it is threshold-free and rewards
#    ranking the rare stroke cases well, which is what we want under 6% positives.
# ----------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
param_grid = [
    {"penalty": ["l2"], "C": np.logspace(-3, 2, 11),
     "class_weight": [None, "balanced"], "solver": ["lbfgs"]},
    {"penalty": ["l1"], "C": np.logspace(-3, 2, 11),
     "class_weight": [None, "balanced"], "solver": ["liblinear"]},
]
grid = GridSearchCV(LogisticRegression(max_iter=5000), param_grid, cv=cv,
                    scoring="average_precision", n_jobs=-1, refit=True)
grid.fit(X_tr, y_tr)
best = grid.best_estimator_
print("\nBest params:", grid.best_params_)
print(f"Best CV PR-AUC: {grid.best_score_:.4f}")

# Also report CV ROC-AUC for the best setting (a second view of the same model)
from sklearn.model_selection import cross_val_score
cv_auc = cross_val_score(best, X_tr, y_tr, cv=cv, scoring="roc_auc")
print(f"CV ROC-AUC of best model: {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")

p_va = best.predict_proba(X_va)[:, 1]
rows["LogReg (tuned, thr 0.50)"] = metrics(y_va, p_va)

# ----------------------------------------------------------------------------
# 4. Threshold selection on VALIDATION only
#    Goal: catch most strokes while keeping false alarms tolerable.
#    Rule: highest F2 (recall weighted 2x precision). A missed stroke is more
#    costly than a false alarm in a screening aid. Change BETA to shift tradeoff.
# ----------------------------------------------------------------------------
BETA = 2.0
prec, rec, thr = precision_recall_curve(y_va, p_va)
prec, rec = prec[:-1], rec[:-1]
fbeta = (1 + BETA**2) * prec * rec / np.clip(BETA**2 * prec + rec, 1e-12, None)
best_thr = float(thr[np.argmax(fbeta)])
print(f"\nThreshold chosen on validation (max F{BETA:g}): {best_thr:.3f}")
rows[f"LogReg (tuned, thr {best_thr:.2f})"] = metrics(y_va, p_va, best_thr)

val_table = pd.DataFrame(rows).T
print("\n=== VALIDATION RESULTS ===")
print(val_table.round(3).to_string())
val_table.round(4).to_csv(OUT_DIR / "validation_results.csv")

# ----------------------------------------------------------------------------
# 5. FINAL single evaluation on TEST
# ----------------------------------------------------------------------------
p_te = best.predict_proba(X_te)[:, 1]
test_default = metrics(y_te, p_te, 0.5)
test_final = metrics(y_te, p_te, best_thr)
test_table = pd.DataFrame({"tuned @ 0.50": test_default,
                           f"tuned @ {best_thr:.2f}": test_final}).T
dm = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
print("\n=== TEST RESULTS (one-time) ===")
print(test_table.round(3).to_string())
print(f"Majority-class test accuracy: {accuracy_score(y_te, dm.predict(X_te)):.3f}")
test_table.round(4).to_csv(OUT_DIR / "test_results.csv")

pred_te = (p_te >= best_thr).astype(int)
print("\nClassification report (test, chosen threshold):")
print(classification_report(y_te, pred_te, target_names=["no stroke", "stroke"]))
cm = confusion_matrix(y_te, pred_te)
print("Confusion matrix [[TN FP],[FN TP]]:\n", cm)

# Bootstrap CI on test metrics: with 37 test strokes, single numbers are noisy
rng = np.random.default_rng(SEED)
boot = {k: [] for k in ["recall", "precision", "roc_auc"]}
for _ in range(1000):
    idx = rng.integers(0, len(y_te), len(y_te))
    if y_te[idx].sum() == 0:
        continue
    pr = (p_te[idx] >= best_thr).astype(int)
    boot["recall"].append(recall_score(y_te[idx], pr))
    boot["precision"].append(precision_score(y_te[idx], pr, zero_division=0))
    boot["roc_auc"].append(roc_auc_score(y_te[idx], p_te[idx]))
ci = {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in boot.items()}
print("\n95% bootstrap CIs on test:")
for k, (lo, hi) in ci.items():
    print(f"  {k}: {lo:.3f} to {hi:.3f}")

# ----------------------------------------------------------------------------
# 6. Explainability
# ----------------------------------------------------------------------------
coef = pd.DataFrame({"feature": features, "coef": best.coef_[0]})
coef["odds_ratio_per_unit"] = np.exp(coef["coef"])
coef = coef.reindex(coef["coef"].abs().sort_values(ascending=False).index)
print("\n=== COEFFICIENTS (scaled features: 1 unit = 1 std for age/glucose/bmi) ===")
print(coef.round(3).to_string(index=False))
coef.round(4).to_csv(OUT_DIR / "coefficients.csv", index=False)

# Worked example: highest-risk and a median-risk test patient
order = np.argsort(p_te)
ex_idx = [order[-1], order[len(order) // 2]]
print("\n=== WORKED EXAMPLES (test set) ===")
for i in ex_idx:
    contrib = (X_te.iloc[i].to_numpy() * best.coef_[0])
    top = pd.Series(contrib, index=features).sort_values(key=np.abs, ascending=False).head(4)
    print(f"original_row {r_te[i]}: predicted risk {p_te[i]:.3f}, actual stroke={y_te[i]}")
    print("  intercept:", round(float(best.intercept_[0]), 3))
    print("  top contributions to log-odds:", top.round(3).to_dict())

# ----------------------------------------------------------------------------
# 7. Optional check: extreme BMI (> 60) in or out
# ----------------------------------------------------------------------------
cleaned = pd.read_csv(DATA_DIR / "stroke_cleaned.csv").set_index("original_row")
bmi_raw_tr = cleaned.loc[r_tr, "bmi"].to_numpy()
bmi_raw_te = cleaned.loc[r_te, "bmi"].to_numpy()
keep_tr = bmi_raw_tr <= 60
keep_te = bmi_raw_te <= 60
m60 = LogisticRegression(max_iter=5000, **{k: v for k, v in grid.best_params_.items()}).fit(
    X_tr[keep_tr], y_tr[keep_tr])
p60 = m60.predict_proba(X_te[keep_te])[:, 1]
print(f"\n=== BMI > 60 check === rows dropped: train {(~keep_tr).sum()}, test {(~keep_te).sum()}")
print(f"ROC-AUC on test, all rows: {roc_auc_score(y_te, p_te):.4f}")
print(f"ROC-AUC on test, BMI<=60 model+rows: {roc_auc_score(y_te[keep_te], p60):.4f}")
print(f"ROC-AUC on test BMI<=60 rows, original model: {roc_auc_score(y_te[keep_te], p_te[keep_te]):.4f}")

# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
fpr, tpr, _ = roc_curve(y_te, p_te)
ax[0].plot(fpr, tpr, label=f"Tuned LogReg (AUC {roc_auc_score(y_te, p_te):.3f})")
ax[0].plot([0, 1], [0, 1], "k--", label="Chance")
ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test)")
ax[0].legend()
pr, rc, _ = precision_recall_curve(y_te, p_te)
ax[1].plot(rc, pr, label=f"AP {average_precision_score(y_te, p_te):.3f}")
ax[1].axhline(y_te.mean(), color="k", ls="--", label=f"Base rate {y_te.mean():.3f}")
ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (test)")
ax[1].legend()
plt.tight_layout(); plt.savefig(OUT_DIR / "roc_pr_curves.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay(cm, display_labels=["no stroke", "stroke"]).plot(ax=ax, colorbar=False)
ax.set_title(f"Confusion matrix (test, thr {best_thr:.2f})")
plt.tight_layout(); plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150); plt.close()

c = coef.sort_values("coef")
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.barh(c["feature"], c["coef"], color=np.where(c["coef"] > 0, "#c0392b", "#2c7fb8"))
ax.axvline(0, color="k", lw=0.8)
ax.set(xlabel="Coefficient (log-odds)", title="Logistic regression coefficients")
plt.tight_layout(); plt.savefig(OUT_DIR / "coefficients.png", dpi=150); plt.close()

# Save machine-readable summary
summary = {
    "best_params": {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                    for k, v in grid.best_params_.items()},
    "cv_pr_auc": float(grid.best_score_),
    "cv_roc_auc_mean": float(cv_auc.mean()),
    "threshold": best_thr,
    "test_default": test_default, "test_final": test_final,
    "test_confusion_matrix": cm.tolist(),
    "bootstrap_ci": {k: [float(a), float(b)] for k, (a, b) in ci.items()},
}
(OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
print(f"\nSaved tables and figures to {OUT_DIR.resolve()}")
