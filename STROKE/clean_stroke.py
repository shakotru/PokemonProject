import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.makedirs("cleaned", exist_ok=True)

# 1-3. Drop id, the single "Other" gender row, and minors
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df = df.drop(columns="id")
df = df[df["gender"] != "Other"]
df = df[df["age"] >= 18]

# 4. Flag missing BMI, build age groups for filling (no BMI outliers removed)
df["bmi_missing"] = df["bmi"].isna().astype(int)
df["age_group"] = pd.cut(df["age"], [17, 40, 60, 120], labels=["18-40", "41-60", "60+"]).astype(str)

# Split first (70/15/15, stratified so each part keeps ~5.8% strokes)
X = df.drop(columns="stroke")
y = df["stroke"]
X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.3, random_state=101, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_other, y_other, test_size=0.5, random_state=101, stratify=y_other)

# 4 (cont). Fill BMI with the TRAINING average for each age group + gender
bmi_means = X_train.groupby(["age_group", "gender"])["bmi"].mean().to_dict()

def fill_bmi(part):
    part = part.copy()
    keys = pd.Series(list(zip(part["age_group"], part["gender"])), index=part.index)
    part["bmi"] = part["bmi"].fillna(keys.map(bmi_means))
    return part

X_train, X_val, X_test = fill_bmi(X_train), fill_bmi(X_val), fill_bmi(X_test)

# 5-6. Keep "Unknown" smoking as a category; turn text columns into 0/1 columns
X_train, X_val, X_test = [pd.get_dummies(p.drop(columns="age_group"), drop_first=True) for p in (X_train, X_val, X_test)]
X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Save the cleaned dataset (unscaled, easy to read): one file with a "split" column
full = pd.concat([
    X_train.assign(stroke=y_train, split="train"),
    X_val.assign(stroke=y_val, split="validation"),
    X_test.assign(stroke=y_test, split="test"),
]).sort_index()
full.to_csv("cleaned/stroke_cleaned.csv", index_label="original_row")

# 7. Scale numeric columns (fit on training data only)
num_cols = ["age", "avg_glucose_level", "bmi"]
scaler = StandardScaler().fit(X_train[num_cols])
for name, part, target in [("train", X_train, y_train), ("validation", X_val, y_val), ("test", X_test, y_test)]:
    part[num_cols] = scaler.transform(part[num_cols])
    part.assign(stroke=target).to_csv(f"cleaned/stroke_{name}_scaled.csv", index_label="original_row")

print("Rows:", len(X_train), len(X_val), len(X_test))
print("Strokes:", y_train.sum(), y_val.sum(), y_test.sum())
print("Missing values left:", X_train.isna().sum().sum(), X_val.isna().sum().sum(), X_test.isna().sum().sum())
print("Saved files in cleaned/:", sorted(os.listdir("cleaned")))
