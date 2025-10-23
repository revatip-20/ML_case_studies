# wine_classifier.py
# Requirements: pandas, numpy, scikit-learn, seaborn, matplotlib, joblib
# Install if needed: pip install pandas numpy scikit-learn seaborn matplotlib joblib

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc
)
import joblib
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print("Data shape:", df.shape)
print(df.head())

# --------------------------------------------------
# 2. Basic exploration
# --------------------------------------------------
print("\nTarget classes:", dict(zip(range(len(data.target_names)), data.target_names)))

plt.figure(figsize=(6,4))
sns.countplot(x='target', data=df)
plt.title("Wine Class Distribution")
plt.show()

print("\nSummary statistics:")
print(df.describe())

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
# 5. Train baseline models
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

# --------------------------------------------------
# 6. Hyperparameter tuning for Random Forest
# --------------------------------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 6, 10],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=4, scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(X_train_scaled, y_train)
print("\nBest RF params:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

best_model = grid.best_estimator_

# --------------------------------------------------
# 7. Evaluation on test set
# --------------------------------------------------
y_pred = best_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print("\nFinal Random Forest Model Performance:")
print(f"Test Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# --------------------------------------------------
# 8. Feature Importance
# --------------------------------------------------
feat_imp = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)[:15]
print("\nTop feature importances:\n", feat_imp)
plt.figure(figsize=(7,4))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Top 15 Important Features (Random Forest)")
plt.show()

# --------------------------------------------------
# 9. ROC Curves (multi-class)
# --------------------------------------------------
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_proba = best_model.predict_proba(X_test_scaled)

fpr = {}
tpr = {}
roc_auc = {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(7,5))
for i, name in enumerate(data.target_names):
    plt.plot(fpr[i], tpr[i], label=f'{name} (AUC = {roc_auc[i]:.3f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC Curves (Random Forest)')
plt.legend(loc='lower right')
plt.show()

# --------------------------------------------------
# 10. Save model and scaler
# --------------------------------------------------
joblib.dump(best_model, "wine_rf_model.joblib")
joblib.dump(scaler, "scaler.joblib")
print("✅ Model and scaler saved successfully!")

# --------------------------------------------------
# 11. Example prediction for a new wine sample
# --------------------------------------------------
example = pd.DataFrame([X.iloc[0]])  # Example using first wine sample
example_scaled = scaler.transform(example)
pred = best_model.predict(example_scaled)[0]
proba = best_model.predict_proba(example_scaled)[0]

print(f"\nPredicted Wine Class: {data.target_names[pred]}")
print(f"Class Probabilities: {dict(zip(data.target_names, np.round(proba, 3)))}")
