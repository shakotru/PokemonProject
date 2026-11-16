# Stroke Risk Prediction: Project Overview, Progress and Data Cleaning

Audience: project teammates (human readers and LLMs). Last updated 2026-11-15.
Status tags: [DONE] finished, [TODO] not started, [VERIFY] needs confirmation.

---

## 1. Project summary

- **Course:** Machine Learning course project (applied ML; the announcement is themed around workforce, scheduling, payroll and employee analytics).
- **Our topic:** predict whether a patient has had a stroke from routine health and demographic information. This is a screening-support model, not a diagnostic tool.
- **ML task:** binary classification with severe class imbalance (about 5% positives).
- **Target column:** `stroke` (1 = had a stroke, 0 = did not).
- **Earlier idea (dropped):** a Pokemon "is_legendary" classifier. We replaced it because it had no real-world application and did not fit the course themes.
- **[VERIFY] Topic approval:** the announcement lists workforce topics (under-staffing forecasting, time-off impact, punch exceptions, employee attrition, schedule fairness) but also says teams may propose their own solution design and look for their own datasets. Get explicit approval from the professor for a medical topic before investing further.

### What the assignment requires (from the announcement)

- A simple baseline model plus at least two ML models (example set: majority-class baseline, logistic regression, random forest, gradient boosting).
- Classification metrics: accuracy, precision, recall, F1, confusion matrix, ROC-AUC. Do not report only one metric; explain each in context.
- Required sections: problem understanding; ML problem formulation; dataset description; EDA; feature engineering; model development; evaluation; explainability; business/real-world interpretation; responsible ML and ethics; limitations and future work.
- Deliverables: report or slides, a clean notebook, dataset or dataset description, final recommendation, limitations and future work.
- Rubric weights: Model Development 20; Problem Understanding 15; Dataset and EDA 15; Feature Engineering 15; Evaluation 15; Explainability and Business Value 10; Responsible ML and Limitations 10.
- Professor's constraint: use techniques taught in class (see Section 6).
- Not stated in the announcement: due date, team size, submission format [VERIFY].

---

## 2. The dataset

- **Name:** Stroke Prediction Dataset (Kaggle), file `healthcare-dataset-stroke-data.csv`.
- **Size:** 5,110 rows, 12 columns.
- **Columns:**
  - `id` (identifier, not a feature)
  - `gender` (Female 2,994; Male 2,115; Other 1)
  - `age` (0.08 to 82)
  - `hypertension` (0/1), `heart_disease` (0/1)
  - `ever_married` (Yes/No)
  - `work_type` (Private, Self-employed, children, Govt_job, Never_worked)
  - `Residence_type` (Urban/Rural)
  - `avg_glucose_level` (55.12 to 271.74)
  - `bmi` (10.3 to 97.6; 201 missing)
  - `smoking_status` (never smoked, Unknown, formerly smoked, smokes)
  - `stroke` (target)
- **Class balance:** 249 strokes (4.87%) versus 4,861 non-strokes.
- **No duplicates:** no duplicate ids or duplicate rows.
- **[VERIFY] Provenance:** we could not confirm how the data was collected or whether it is fully real. State this clearly in the limitations section.
- **Not used in class:** a search of the course notebooks found no stroke dataset, so this is not a repeat of a class example. (Class datasets include a hearing-test set, Iris, a heart disease set, Advertising, Palmer Penguins, banknote authentication and Ames Housing.)

### Key findings from exploration

| Finding | Detail | Why it matters |
|---|---|---|
| Class imbalance | Always predicting "no stroke" scores about 95% accuracy | Accuracy is misleading; recall and precision for the stroke class are essential |
| Age dominates | Stroke rate is 0.4% for ages up to 40, 4.1% for 41-60, 8.2% for 61-70, 18.0% for over 70 | Age alone gave a ROC-AUC of about 0.864, nearly the full model's 0.866 |
| Missing BMI is informative | 201 rows missing BMI; stroke rate 19.9% for those rows versus 4.3% for the rest (about 40 strokes among them) | Dropping them would lose 16% of all positive cases; the missingness itself is a signal |
| Minors | 856 patients under 18, only 2 strokes | They dilute the data and create odd categories (`children` work type, mostly Unknown smoking) |
| Unknown smoking | 1,544 rows (30%), including 682 of the 856 minors | Treated as its own category rather than discarded |
| High glucose is real signal | 434 rows above 200 have a 12.9% stroke rate | Do not remove high glucose values |
| Extreme BMI | Max 97.6 (a 17-year-old); max 92 among adults | Rare but medically possible; we decided to keep all BMI values |

Exploratory model results (done before cleaning, on the full data with a 70/30 split, so treat them as rough):

| Model | Accuracy | Stroke precision | Stroke recall | ROC-AUC |
|---|---|---|---|---|
| Majority class | 0.951 | n/a | 0 | n/a |
| Logistic regression | 0.948 | 0.14 | 0.01 | 0.866 |
| Logistic regression, balanced class weights | 0.755 | 0.15 | 0.85 | 0.864 |
| Random forest, balanced | 0.922 | 0.23 | 0.25 | 0.847 |

Takeaway: a default model almost never catches strokes. Balancing class weights catches most strokes at the cost of many false alarms. This tradeoff is the core of our evaluation discussion.

---

## 3. Progress so far

- [DONE] Chose the stroke topic and dataset (after dropping the Pokemon idea).
- [DONE] Compared candidate datasets and risks (class imbalance, missing BMI, minors, unknown smoking).
- [DONE] Reviewed which course techniques apply (Section 6).
- [DONE] Wrote the data cleaning script `clean_stroke.py` (Section 4), run successfully on the real file.
- [DONE] Produced cleaned CSV outputs in a `cleaned/` folder:
  - `stroke_cleaned.csv`: full cleaned dataset, unscaled, with a `split` column (train/validation/test)
  - `stroke_train_scaled.csv`, `stroke_validation_scaled.csv`, `stroke_test_scaled.csv`: scaled versions for modeling
- [TODO] Professor approval of the topic.
- [DONE] Baseline, tuned logistic regression, threshold selection, test evaluation, explainability, BMI>60 check (`stroke_model.py`, Section 9).
- [TODO] EDA charts, decision tree / random forest comparison, write-up (Section 7).

---

## 4. The data cleaning script (`clean_stroke.py`)

Run it with `py clean_stroke.py` from the folder containing `healthcare-dataset-stroke-data.csv`. It never modifies the raw file.

### Final output of the script

- 4,253 rows after cleaning: 2,977 train, 638 validation, 638 test.
- Strokes per split: 173 train, 37 validation, 37 test (247 total).
- No missing values remain. The maximum BMI of 92 is still present.

### Step-by-step with justification

**Step 1. Drop `id`.**
- It is an identifier with no predictive meaning. Keeping it could let a model memorize rows or find accidental patterns.

**Step 2. Drop the one row with `gender = Other`.**
- A single row cannot teach a model anything about a category and would create a dummy column that is zero everywhere else. With only one example, it also cannot appear in both training and test sets.
- Cost: one row, negligible.

**Step 3. Keep adults only (`age >= 18`).**
- Stroke screening targets adults; only 2 of 856 minors had strokes, so including them mostly teaches the model that "children rarely have strokes".
- It removes oddities that exist only because of children: the `children` work type, many Unknown smoking entries (682 of 856 minors), and fractional infant ages (43 patients under age 1).
- This matches the course's outlier notebook, which trims data using domain knowledge (its example drops anyone under voting age).
- Cost: 856 rows and 2 strokes. Remaining: 4,253 rows with 247 strokes (5.8%).
- This is a team decision that must be stated in the report. It narrows the claim to "adult stroke risk".

**Step 4. Add `bmi_missing` flag, then fill missing BMI with an average by age group and gender.**
- We keep the rows because dropping them would lose about 40 positive cases, which is a large share of an already tiny positive class.
- The flag preserves the signal that "BMI was not recorded", which correlates with stroke (19.9% versus 4.3%).
- Group averages (age groups 18-40, 41-60, 60+, split by gender) beat one overall average because BMI varies by age. The missing-BMI rows skew older (median age 58).
- This follows the course's missing data notebook, which fills missing values using `groupby(...).transform(...)` averages.
- The group averages are computed from the training split only, so no information from validation or test data leaks into the model. (The class notebook computes them on the whole dataset; our version is stricter.)

**Step 5. Keep "Unknown" smoking status as its own category.**
- Dropping 30% of the rows would hurt more than it helps. "Unknown" may mean different things (not asked, not recorded), and it has a distinct stroke rate (3.0%), so the model can use it.
- Interpretation caveat: do not describe it as "non-smokers".

**Step 6. Convert text columns to 0/1 columns (dummy encoding).** See Section 5.

**Step 7. Do not remove or cap BMI outliers.**
- BMI values like 92 or 97.6 are rare but medically possible, and the course's outlier notebook warns that an outlier is still a data point and should only be removed with good reason.
- Nothing in the data shows these values are errors. The 97.6 value belongs to a 17-year-old and is gone only because of the adults-only rule (Step 3), not an outlier rule.
- Optional later check: run models with and without BMI values above about 60 and report whether results change.

**Step 8. Do not remove high glucose values.**
- Glucose above 200 had a 12.9% stroke rate, far above average, so these values carry signal.

**Step 9. Split into train, validation and test (70/15/15) with stratification.** See Section 5.
- `random_state=101` matches the course notebooks, and everyone gets identical splits.
- The split is done before filling BMI and before scaling so that statistics come from training data only.

**Step 10. Scale `age`, `avg_glucose_level` and `bmi` with `StandardScaler`.**
- The three columns have very different ranges (age 18-82, glucose 55-272, BMI 11-92). Logistic regression is sensitive to that, and regularization penalizes coefficients unevenly if scales differ.
- The scaler is fit on training data only, then applied to validation and test data, as in the course's cross-validation notebook.
- The 0/1 columns are not scaled.
- Tree models (decision tree, random forest) do not need scaling; use the unscaled file for them if you prefer.

### Output files

- `cleaned/stroke_cleaned.csv`: human-readable cleaned data with a `split` column. Use for EDA and charts.
- `cleaned/stroke_*_scaled.csv`: scaled train/validation/test files for logistic regression. Each includes the `stroke` target and an `original_row` index that maps back to the raw file.

---

## 5. Dummy encoding and stratification explained

### Dummy encoding (`pd.get_dummies`)

**Problem:** models need numbers, but columns like `smoking_status` and `work_type` contain text. Converting categories to 1, 2, 3 would imply a false order (for example, that "smokes" is three times "formerly smoked").

**Solution:** create one 0/1 column per category. Example for `smoking_status`:

| Original value | `never smoked` | `smokes` |
|---|---|---|
| formerly smoked | 0 | 0 |
| never smoked | 1 | 0 |
| smokes | 0 | 1 |

**Why `drop_first=True`:** one category is dropped; a row of all zeros means that category, so no information is lost. Keeping all columns makes them perfectly redundant (they always add up to 1), which is the "dummy variable trap" and confuses logistic regression coefficients. This is covered in the course's categorical data notebook.

**Resulting columns in our data (15 features):** `age`, `hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`, `bmi_missing`, `gender_Male`, `ever_married_Yes`, `work_type_Never_worked`, `work_type_Private`, `work_type_Self-employed`, `Residence_type_Urban`, `smoking_status_formerly smoked`, `smoking_status_never smoked`, `smoking_status_smokes`.

**Implementation detail:** the script encodes each split and then aligns validation and test columns to the training columns, so all three sets have identical columns.

### Stratification

**Problem:** only about 5.8% of adult patients had a stroke. A plain random split can accidentally give one part a different stroke rate. In our check, a plain split gave 5.5% strokes in training and 6.5% in the test set (83 test strokes), while a stratified split gave 5.8% in both (74 test strokes).

**Solution:** `train_test_split(..., stratify=y)` forces every split to keep the same stroke percentage as the whole dataset.

**Why it matters here:**
- With only 247 positive cases, one extra or missing stroke in a split noticeably shifts precision and recall.
- Comparable splits make model comparisons fair and results more repeatable.

**How we apply it:** first split 70% train / 30% other (stratified), then split "other" 50/50 into validation and test (stratified again). The result is 15% each.

**Remaining limitation:** validation and test each contain only 37 strokes, so metrics from a single split are noisy. Use `cross_val_score` or `GridSearchCV` on the training data to tune and compare models, and keep the test set for one final evaluation.

---

## 6. Course techniques we are using

- Missing data handling with group averages (`groupby` + `transform`): missing data notebook.
- Dummy variables with `drop_first=True`: categorical data notebook.
- Outlier handling guidance and domain-knowledge trimming: outliers notebook.
- Train/validation/test split, `StandardScaler` fit on train only, `cross_val_score`: cross-validation notebook.
- `GridSearchCV` for tuning: grid search notebook.
- Logistic regression, classification report, confusion matrix, ROC and precision-recall curves: logistic regression notebooks.
- Decision trees and random forests with `feature_importances_`: tree notebooks.
- Also available: AdaBoost, gradient boosting, SVM, PCA, regularization.

Not in the class notebooks (explain if used): `stratify=`, `class_weight="balanced"`, and a `bmi_missing` indicator column. These are standard scikit-learn/pandas options, but mention them as additions.

---

## 7. What needs to happen next

1. [TODO] Get professor approval for the medical topic and confirm due date and submission format.
2. [TODO] Load `stroke_cleaned.csv` and make EDA charts: stroke rate by age group, class balance, missing BMI versus stroke, glucose and BMI boxplots by outcome, correlation heatmap, smoking status versus stroke.
3. [DONE] Baseline: majority-class accuracy (about 94.2% on adults) as the reference to beat.
4. [TODO] Models: logistic regression (plain and `class_weight="balanced"`), decision tree, random forest, optionally gradient boosting. Compare on identical splits.
5. [DONE] (logistic regression) Tune with `GridSearchCV` (for example `scoring="recall"` or `"f1"`) and 5-fold cross-validation on the training data.
6. [DONE] (logistic regression) Evaluate: precision, recall, F1 for the stroke class, confusion matrix, ROC-AUC, precision-recall curve. Discuss the recall versus precision tradeoff (missing a stroke versus false alarms).
7. [DONE] (logistic regression coefficients) Explainability: logistic regression coefficients, random forest feature importances, optionally permutation importance; one worked example prediction.
8. [DONE] (logistic regression) Final single evaluation on the test set after choosing the model with validation/cross-validation only.
9. [DONE] Optional check: results with and without extreme BMI values.
10. [TODO] Write-up following the announcement's sections, including real-world use (screening aid that flags adults for follow-up) and responsible ML.
11. [TODO] Clean the notebook deliverable and split the work between teammates.

### Points to cover in the responsible-ML and limitations sections

- It is a screening aid, never a diagnosis; false negatives (missed strokes) and false positives (unnecessary worry and testing) both have costs.
- Age is the main driver, so the model may just be flagging older patients; discuss the added value of other features.
- Unknown data origin, small number of positive cases, one-time snapshot without follow-up, possible bias in who is in the dataset.
- Privacy: the data is de-identified, and no personal identifiers should be added.
- Adults-only scope, imputation of BMI, and "Unknown" smoking category are assumptions that affect conclusions.

---

## 8. Working rules for the team

- Use `py` (not `python`) in PowerShell on Windows.
- Always use the provided split (`random_state=101`) so results are comparable.
- Fit any preprocessing (imputation averages, scalers) on training data only.
- Do not tune against the test set.
- Do not use any column derived from `stroke` as an input.

---

## 9. Logistic regression results (`stroke_model.py`)

**Note on terminology:** the target `stroke` is 0/1, so the linear model used is logistic regression (linear regression would give predictions outside 0 to 1 and is not suited to a binary target). Both are linear models; logistic regression passes the linear score through a sigmoid to produce a probability.

### Method
- Data: provided scaled splits (2,977 train / 638 validation / 638 test, 173 / 37 / 37 strokes). `original_row` is not used as a feature.
- Tuning: `GridSearchCV`, 5-fold stratified CV on the training data only. Grid: L1 and L2 penalty, 11 values of C from 0.001 to 100, class weight none or balanced. Scoring: average precision (PR-AUC), because it rewards ranking the rare stroke cases well and does not depend on a threshold.
- Best setting: L1 penalty, C = 3.16, no class weights. CV PR-AUC 0.237; CV ROC-AUC 0.812 (+/- 0.038).
- Threshold: chosen on the validation set only by maximizing F2 (recall weighted twice as much as precision, since a missed stroke costs more than a false alarm). Chosen threshold: 0.134.
- Test set: evaluated once, after the model and threshold were fixed.

### Validation results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Majority class | 0.942 | 0 | 0 | 0 | 0.500 | 0.058 |
| LogReg plain, threshold 0.50 | 0.940 | 0 | 0 | 0 | 0.831 | 0.226 |
| LogReg balanced, threshold 0.50 | 0.721 | 0.146 | 0.784 | 0.246 | 0.836 | 0.282 |
| LogReg tuned, threshold 0.50 | 0.939 | 0 | 0 | 0 | 0.832 | 0.233 |
| LogReg tuned, threshold 0.134 | 0.864 | 0.234 | 0.595 | 0.336 | 0.832 | 0.233 |

### Test results (single evaluation, tuned model, threshold 0.134)

| Metric | Value | 95% bootstrap interval |
|---|---|---|
| Accuracy | 0.868 | |
| Precision (stroke) | 0.247 | 0.165 to 0.333 |
| Recall (stroke) | 0.622 | 0.463 to 0.788 |
| F1 (stroke) | 0.354 | |
| ROC-AUC | 0.850 | 0.793 to 0.902 |
| PR-AUC | 0.234 | |

Confusion matrix on test (rows actual, columns predicted): 531 true negatives, 70 false positives, 14 false negatives, 23 true positives. The majority-class baseline scores 0.942 accuracy on test but catches no strokes.

### Interpretation
- At the default 0.5 threshold the model predicts "no stroke" for everyone (precision and recall are 0). Probabilities for a 5.8% event rarely exceed 0.5, so the threshold, not the model's ranking ability, is the problem. ROC-AUC (0.85) shows the ranking is good.
- Lowering the threshold to 0.134 catches 23 of 37 strokes (62%) at the cost of 70 false alarms. Accuracy drops from 0.94 to 0.87, which is expected and acceptable for a screening aid. Accuracy should not be used to judge this model.
- Precision is about 25%: roughly 1 in 4 flagged adults had a stroke, versus a base rate of 5.8%, so flagged patients are about 4 times more likely than average to be positive.
- With only 37 test strokes, the intervals are wide (recall 0.46 to 0.79). Differences of a few percentage points between models are not meaningful.
- Balanced class weights reach higher recall at 0.50 (0.78) but with lower precision and accuracy; after threshold tuning the two approaches rank cases about equally well (validation ROC-AUC 0.836 vs 0.832).

### Explainability (coefficients; age, glucose and BMI are in standard deviations)

| Feature | Coefficient | Odds ratio per unit |
|---|---|---|
| age | 1.278 | 3.59 per SD |
| bmi_missing | 1.041 | 2.83 |
| ever_married_Yes | -0.360 | 0.70 |
| hypertension | 0.283 | 1.33 |
| avg_glucose_level | 0.269 | 1.31 per SD |
| work_type_Private | 0.251 | 1.29 |
| gender_Male | 0.178 | 1.20 |
| smoking_status_smokes | 0.075 | 1.08 |
| bmi | 0.061 | 1.06 per SD |

`heart_disease`, `Residence_type_Urban` and `work_type_Never_worked` were set to 0 by the L1 penalty. Smaller effects (self-employed, formerly smoked, never smoked) are near zero.

- Age is the dominant driver, consistent with the EDA finding (age alone gave an AUC near the full model's).
- `bmi_missing` is the second-largest effect. This reflects how the data was recorded, not biology, and should be described that way in the report.
- `ever_married_Yes` has a negative sign after accounting for age. Age and marital status are strongly related, so treat this as a statistical artifact, not a causal effect. The same caution applies to `work_type_Private` and `gender_Male`.
- `heart_disease` being dropped by L1 does not mean it is unimportant; its information overlaps with age and hypertension in this data. Also note: coefficients are association, not causation.
- Worked example (test row 4164): intercept -3.555, plus age +2.297, bmi_missing +1.041, glucose +0.645, married -0.360, giving a predicted risk of 0.63 (this patient did not have a stroke, so it is a false positive). A low-risk example (row 2976) has predicted risk 0.024.

### Checks and notes
- **Extreme BMI (> 60):** removing 9 training rows and 1 test row changed test ROC-AUC from 0.8495 to 0.8492. No meaningful effect, so keeping all BMI values is justified.
- **Extra terms:** squared age/glucose/BMI and age interactions were tried (CV on train, scored on validation). They did not improve CV or validation scores (validation ROC-AUC 0.831 to 0.834 versus 0.832), so the simpler model was kept.
- **Remaining TODO:** compare with decision tree and random forest on the same splits, since the assignment needs at least two ML models beyond the baseline. Logistic regression alone does not satisfy that.
- Files: `model_output/` holds `validation_results.csv`, `test_results.csv`, `coefficients.csv`, `summary.json`, `roc_pr_curves.png`, `confusion_matrix.png`, `coefficients.png`.
