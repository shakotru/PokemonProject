"""
Stroke risk model: decision tree, same splits and protocol as stroke_model.py.

  1. Load provided train / validation / test splits (random_state=101).
  2. Tune with GridSearchCV (5-fold stratified CV on TRAIN only).
  3. Choose the decision threshold on VALIDATION only (max F2).
  4. Evaluate once on TEST; compare with logistic regression.
  5. Feature importances and a plot of the tree.

Trees are unaffected by feature scaling, so the scaled files are used as-is.
Run after stroke_model.py (it reads model_output/summary.json for the comparison).
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")
DATA_DIR = Path(".")
OUT_DIR = Path("model_output")
OUT_DIR.mkdir(exist_ok=True)
SEED = 101
BETA = 2.0


def load(name):
    df = pd.read_csv(DATA_DIR / f"stroke_{name}_scaled.csv")
    y = df["stroke"].astype(int).to_numpy()
    X = df.drop(columns=["original_row", "stroke"]).astype(float)
    return X, y


def metrics(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {"accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, proba),
            "pr_auc": average_precision_score(y, proba), "threshold": thr}


def pick_threshold(y, p, beta=BETA):
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    f = (1 + beta**2) * prec * rec / np.clip(beta**2 * prec + rec, 1e-12, None)
    return float(thr[np.argmax(f)])


X_tr, y_tr = load("train")
X_va, y_va = load("validation")
X_te, y_te = load("test")
features = X_tr.columns.tolist()

# ---------------------------------------------------------------------------
# Tune (train only). Shallow, heavily regularized trees: with 173 strokes an
# unconstrained tree just memorizes the training data.
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
param_grid = {
    "max_depth": [2, 3, 4, 5, 6, 8],
    "min_samples_leaf": [5, 10, 20, 40, 80],
    "ccp_alpha": [0.0, 0.0005, 0.001, 0.002],
    "class_weight": [None, "balanced"],
    "criterion": ["gini", "entropy"],
}
grid = GridSearchCV(DecisionTreeClassifier(random_state=SEED), param_grid, cv=cv,
                    scoring="average_precision", n_jobs=-1, refit=True)
grid.fit(X_tr, y_tr)
tree = grid.best_estimator_
print("Best params:", grid.best_params_)
print(f"Best CV PR-AUC: {grid.best_score_:.4f}")
cv_auc = cross_val_score(tree, X_tr, y_tr, cv=cv, scoring="roc_auc")
print(f"CV ROC-AUC: {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")

# Untuned reference: shows overfitting of an unconstrained tree
deep = DecisionTreeClassifier(random_state=SEED).fit(X_tr, y_tr)

# ---------------------------------------------------------------------------
# Validation + threshold
# ---------------------------------------------------------------------------
p_va = tree.predict_proba(X_va)[:, 1]
thr = pick_threshold(y_va, p_va)
print(f"\nThreshold chosen on validation (max F{BETA:g}): {thr:.3f}")
val_rows = {
    "Tree (unconstrained, thr 0.50)": metrics(y_va, deep.predict_proba(X_va)[:, 1]),
    "Tree (tuned, thr 0.50)": metrics(y_va, p_va),
    f"Tree (tuned, thr {thr:.2f})": metrics(y_va, p_va, thr),
}
val_table = pd.DataFrame(val_rows).T
print("\n=== VALIDATION ===")
print(val_table.round(3).to_string())
print(f"Unconstrained tree train accuracy: {deep.score(X_tr, y_tr):.3f} "
      f"(depth {deep.get_depth()}, {deep.get_n_leaves()} leaves)")
val_table.round(4).to_csv(OUT_DIR / "tree_validation_results.csv")

# ---------------------------------------------------------------------------
# Test (one time)
# ---------------------------------------------------------------------------
p_te = tree.predict_proba(X_te)[:, 1]
test_final = metrics(y_te, p_te, thr)
pred = (p_te >= thr).astype(int)
cm = confusion_matrix(y_te, pred)
print("\n=== TEST (one-time) ===")
print(pd.Series(test_final).round(3).to_string())
print("Confusion matrix [[TN FP],[FN TP]]:\n", cm)

rng = np.random.default_rng(SEED)
boot = {k: [] for k in ["recall", "precision", "roc_auc"]}
for _ in range(1000):
    idx = rng.integers(0, len(y_te), len(y_te))
    if y_te[idx].sum() == 0:
        continue
    pr = (p_te[idx] >= thr).astype(int)
    boot["recall"].append(recall_score(y_te[idx], pr))
    boot["precision"].append(precision_score(y_te[idx], pr, zero_division=0))
    boot["roc_auc"].append(roc_auc_score(y_te[idx], p_te[idx]))
ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in boot.items()}
print("95% bootstrap CIs:", {k: tuple(round(x, 3) for x in v) for k, v in ci.items()})

# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
imp = pd.DataFrame({"feature": features, "importance": tree.feature_importances_})
imp = imp.sort_values("importance", ascending=False)
print("\n=== FEATURE IMPORTANCES (impurity) ===")
print(imp[imp.importance > 0].round(3).to_string(index=False))
imp.round(4).to_csv(OUT_DIR / "tree_importances.csv", index=False)

perm = permutation_importance(tree, X_va, y_va, scoring="roc_auc", n_repeats=20,
                              random_state=SEED)
pimp = pd.DataFrame({"feature": features, "perm_auc_drop": perm.importances_mean}
                    ).sort_values("perm_auc_drop", ascending=False)
print("\nPermutation importance on validation (drop in ROC-AUC):")
print(pimp.head(6).round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# Comparison with logistic regression (same splits, same protocol)
# ---------------------------------------------------------------------------
lr_path = OUT_DIR / "summary.json"
if lr_path.exists():
    s = json.loads(lr_path.read_text())
    bp = s["best_params"]
    lr = LogisticRegression(max_iter=5000, **bp).fit(X_tr, y_tr)
    p_lr_va = lr.predict_proba(X_va)[:, 1]
    p_lr_te = lr.predict_proba(X_te)[:, 1]
    lr_thr = s["threshold"]
    comp = pd.DataFrame({
        "LogReg (tuned)": metrics(y_te, p_lr_te, lr_thr),
        "Decision tree (tuned)": test_final,
    }).T
    print("\n=== TEST COMPARISON ===")
    print(comp.round(3).to_string())
    comp.round(4).to_csv(OUT_DIR / "model_comparison_test.csv")
    lr_cv = cross_val_score(lr, X_tr, y_tr, cv=cv, scoring="average_precision")
    print(f"CV PR-AUC  LogReg {lr_cv.mean():.4f} vs Tree {grid.best_score_:.4f}")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in [("Logistic regression", p_lr_te), ("Decision tree", p_te)]:
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_te, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="ROC on test: model comparison")
    ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(OUT_DIR / "roc_comparison.png", dpi=150); plt.close()

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 8))
plot_tree(tree, feature_names=features, class_names=["no stroke", "stroke"],
          filled=True, rounded=True, impurity=False, proportion=False,
          fontsize=8, ax=ax, max_depth=4)
plt.tight_layout(); plt.savefig(OUT_DIR / "decision_tree.png", dpi=130); plt.close()

top = imp[imp.importance > 0].sort_values("importance")
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.barh(top["feature"], top["importance"], color="#2c7fb8")
ax.set(xlabel="Importance", title="Decision tree feature importances")
plt.tight_layout(); plt.savefig(OUT_DIR / "tree_importances.png", dpi=150); plt.close()

(OUT_DIR / "tree_summary.json").write_text(json.dumps({
    "best_params": {k: (v if isinstance(v, (str, type(None))) else float(v))
                    for k, v in grid.best_params_.items()},
    "cv_pr_auc": float(grid.best_score_), "cv_roc_auc_mean": float(cv_auc.mean()),
    "threshold": thr, "test": test_final, "confusion_matrix": cm.tolist(),
    "bootstrap_ci": ci, "depth": int(tree.get_depth()),
    "leaves": int(tree.get_n_leaves())}, indent=2, default=float))
print(f"\nSaved to {OUT_DIR.resolve()}")
