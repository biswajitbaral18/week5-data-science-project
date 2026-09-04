"""
Week 5 Data Science Project
Customer Churn Prediction & Retention Strategy

This project uses a synthetic customer dataset so the workflow is reproducible
when the student's original Week 1-4 dataset is unavailable.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# 1. Create synthetic dataset
np.random.seed(42)
n = 1200

df = pd.DataFrame({
    "Age": np.clip(np.random.normal(38, 12, n), 18, 75).round().astype(int),
    "TenureMonths": np.clip(np.random.gamma(2.2, 12, n), 1, 72).round().astype(int),
    "MonthlyCharge": np.clip(np.random.normal(68, 22, n), 20, 150).round(2),
    "SupportTickets": np.clip(np.random.poisson(1.8, n), 0, 10),
    "LatePayments": np.clip(np.random.poisson(0.7, n), 0, 6),
    "MonthlyUsageHours": np.clip(np.random.normal(24, 8, n), 2, 60).round(1),
    "Contract": np.random.choice(["Monthly", "One Year", "Two Year"], n, p=[.55,.28,.17]),
    "AutoPay": np.random.choice(["Yes", "No"], n, p=[.62,.38]),
    "InternetService": np.random.choice(["Fiber", "DSL", "Cable"], n, p=[.46,.29,.25])
})

logit = (
    -1.55
    + .020*(df["MonthlyCharge"]-60)
    + .22*df["SupportTickets"]
    + .30*df["LatePayments"]
    - .030*df["TenureMonths"]
    + .025*(df["MonthlyUsageHours"]-24)
    + .65*(df["Contract"]=="Monthly")
    + .32*(df["AutoPay"]=="No")
    + .20*(df["InternetService"]=="Fiber")
)
prob = 1/(1+np.exp(-logit))
df["Churn"] = np.random.binomial(1, prob)

# 2. Prepare data
X = df.drop(columns=["Churn"])
y = df["Churn"]

num_cols = [
    "Age", "TenureMonths", "MonthlyCharge",
    "SupportTickets", "LatePayments", "MonthlyUsageHours"
]
cat_cols = ["Contract", "AutoPay", "InternetService"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.20, random_state=42, stratify=y
)

# 3. Train and compare models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8,
        min_samples_leaf=4, random_state=42
    )
}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    score = pipeline.predict_proba(X_test)[:, 1]

    print("\n", name)
    print("Accuracy :", round(accuracy_score(y_test, pred), 3))
    print("Precision:", round(precision_score(y_test, pred, zero_division=0), 3))
    print("Recall   :", round(recall_score(y_test, pred, zero_division=0), 3))
    print("F1       :", round(f1_score(y_test, pred, zero_division=0), 3))
    print("ROC-AUC  :", round(roc_auc_score(y_test, score), 3))

print("\nChurn rate:", round(df["Churn"].mean()*100, 1), "%")
