# breast_cancer_detection.py
# Requirements: pandas, numpy, scikit-learn, seaborn, matplotlib, joblib
# Install if needed: pip install pandas numpy scikit-learn seaborn matplotlib joblib

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc
)
import joblib
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print("Data shape:", df.shape)
print(df.head())

# --------------------------------------------------
# 2. Basic exploration
# --------------------------------------------------
print("\nTarget classes:")
print(dict(zip(data.target_names, np.bincount(data.target))))

plt.figure(figsize=(6,4))
sns.countplot(x='target', data=df)
plt.title("Breast Cancer Diagnosis (0 = Malignant, 1 = Benign)")
plt.show()

print("\nDataset info:")
print(df.info())

# --------------------------------------------------
# 3. Train-test split
# --------------------------------------------------
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# --------------------------------------------------
# 4. Feature scaling
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# 5. Train models
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC AUC:  {roc:.3f}")
    print(classification_report(y_test, y_pred))

# --------------------------------------------------
# 6. Hyperparameter tuning for Random Forest
# --------------------------------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 6, 10],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=4, scoring='roc_auc', n_jobs=-1, verbose=1)
grid.fit(X_train_scaled, y_train)
print("\nBest RF params:", grid.best_params_)
print("Best RF CV ROC AUC:", grid.best_score_)

best_model = grid.best_estimator_

# --------------------------------------------------
# 7. Evaluation on test data
# --------------------------------------------------
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)
print("\nFinal Random Forest Model Performance:")
print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC:  {roc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.title("ROC Curve for Breast Cancer Prediction")
plt.show()

# --------------------------------------------------
# 8. Feature importance
# --------------------------------------------------
feat_imp = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)[:15]
print("\nTop feature importances:\n", feat_imp)
plt.figure(figsize=(7,4))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Top 15 Important Features (Random Forest)")
plt.show()

# --------------------------------------------------
# 9. Save model and scaler
# --------------------------------------------------
joblib.dump(best_model, "breast_cancer_rf_model.joblib")
joblib.dump(scaler, "scaler.joblib")
print("✅ Model and scaler saved successfully!")

# --------------------------------------------------
# 10. Example: predict for a new patient
# --------------------------------------------------
example = pd.DataFrame([X.iloc[0]])  # Use first sample as example
example_scaled = scaler.transform(example)
proba = best_model.predict_proba(example_scaled)[:, 1][0]
print(f"\nExample patient probability of being benign: {proba:.3f}")
