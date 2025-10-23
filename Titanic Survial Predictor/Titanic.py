# titanic_survival_predictor.py
# Requirements: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib
# Install if needed: pip install pandas numpy scikit-learn matplotlib seaborn joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
import joblib
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 1) Load data
# ---------------------------
df = pd.read_csv("train.csv")  # change path if needed
print("Data shape:", df.shape)
print(df.head())

# ---------------------------
# 2) Quick EDA (summary)
# ---------------------------
print("\nMissing values:\n", df.isnull().sum())
print("\nSurvival rate overall:\n", df['Survived'].value_counts(normalize=True))

# Visual: survival by Pclass and Sex (optional)
plt.figure(figsize=(8,4))
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df)
plt.title("Survival rate by Pclass and Sex")
plt.show()

# ---------------------------
# 3) Feature engineering
# ---------------------------
def extract_title(name):
    # Get title from name (Mr, Mrs, Miss, etc.)
    if pd.isnull(name):
        return "Unknown"
    title = name.split(",")[1].split(".")[0].strip()
    return title

df['Title'] = df['Name'].map(extract_title)
# Group rare titles
rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
df['Title'] = df['Title'].replace(rare_titles, 'Rare')

# Family size and is_alone
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Cabin: use only deck letter if present
df['Deck'] = df['Cabin'].fillna('Unknown').map(lambda x: x[0] if x != 'Unknown' else 'Unknown')

# Often Age and Embarked have missing values
# We'll handle these in the pipeline

# Choose features
features = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
    'Embarked', 'Title', 'FamilySize', 'IsAlone', 'Deck'
]
target = 'Survived'

X = df[features].copy()
y = df[target].copy()

# ---------------------------
# 4) Train / test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ---------------------------
# 5) Preprocessing pipelines
# ---------------------------
# Numeric features and pipeline
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Low-cardinality categorical features for one-hot encoding
onehot_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck']
onehot_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Binary features (already numeric)
binary_features = ['IsAlone']
binary_transformer = 'passthrough'

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('ohe', onehot_transformer, onehot_features),
    ('bin', binary_transformer, binary_features)
], remainder='drop')

# ---------------------------
# 6) Models and pipelines
# ---------------------------
# Logistic Regression pipeline
pipe_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])

# Random Forest pipeline
pipe_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
])

# ---------------------------
# 7) Quick cross-validated comparison
# ---------------------------
print("Cross-validating Logistic Regression...")
cv_scores_lr = cross_val_score(pipe_lr, X_train, y_train, cv=5, scoring='roc_auc')
print("Logistic AUC CV:", np.round(cv_scores_lr, 3), "mean:", np.round(cv_scores_lr.mean(), 3))

print("Cross-validating Random Forest...")
cv_scores_rf = cross_val_score(pipe_rf, X_train, y_train, cv=5, scoring='roc_auc')
print("RandomForest AUC CV:", np.round(cv_scores_rf, 3), "mean:", np.round(cv_scores_rf.mean(), 3))

# ---------------------------
# 8) Hyperparameter tuning (Random Forest example)
# ---------------------------
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 6, 10],
    'clf__min_samples_split': [2, 5]
}

grid = GridSearchCV(pipe_rf, param_grid, cv=4, scoring='roc_auc', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
print("Best RF params:", grid.best_params_)
print("Best RF CV AUC:", grid.best_score_)

best_model = grid.best_estimator_

# ---------------------------
# 9) Evaluate on test set
# ---------------------------
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)
print("\nTest Accuracy:", np.round(acc, 4))
print("Test ROC AUC:", np.round(roc, 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# ---------------------------
# 10) Feature importance (for RandomForest)
# ---------------------------
# Get feature names after preprocessing
ohe = best_model.named_steps['preprocessor'].named_transformers_['ohe'].named_steps['onehot']
ohe_feature_names = ohe.get_feature_names_out(onehot_features)
numeric_and_bin = numeric_features + binary_features
feature_names = list(numeric_and_bin) + list(ohe_feature_names)
importances = best_model.named_steps['clf'].feature_importances_

feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
print("\nTop feature importances:\n", feat_imp)
plt.figure(figsize=(6,4))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Top feature importances (RandomForest)")
plt.show()

# ---------------------------
# 11) Save the model
# ---------------------------
joblib.dump(best_model, "titanic_rf_model.joblib")
print("Saved best model to titanic_rf_model.joblib")

# ---------------------------
# 12) Example: predict on a new passenger (dictionary)
# ---------------------------
example = pd.DataFrame([{
    'Pclass': 3, 'Sex': 'male', 'Age': 22, 'SibSp': 1,
    'Parch': 0, 'Fare': 7.25, 'Embarked': 'S', 'Title': 'Mr',
    'FamilySize': 2, 'IsAlone': 0, 'Deck': 'Unknown'
}])
proba = best_model.predict_proba(example)[:,1][0]
print(f"\nExample passenger survival probability: {proba:.3f}")
