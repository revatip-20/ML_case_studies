# Basic Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Text Processing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Ensemble Models
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# For text preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load dataset
df = pd.read_csv("reviews.csv")

# Display first few records
print(df.head())

# Check for missing values
print(df.isnull().sum())

nltk.download('stopwords')
ps = PorterStemmer()

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Apply cleaning
df['Clean_Review'] = df['Review'].apply(clean_text)
print(df[['Review', 'Clean_Review']].head())

# Convert text to numerical vectors
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['Clean_Review']).toarray()
y = df['Sentiment'].map({'Positive': 1, 'Negative': 0})

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("🎯 Random Forest (Bagging) Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

adb = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
adb.fit(X_train, y_train)
y_pred_adb = adb.predict(X_test)

print("⚡ AdaBoost Accuracy:", accuracy_score(y_test, y_pred_adb))

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

print("🔥 Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("🚀 XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))

models = {
    'Random Forest (Bagging)': accuracy_score(y_test, y_pred_rf),
    'AdaBoost': accuracy_score(y_test, y_pred_adb),
    'Gradient Boosting': accuracy_score(y_test, y_pred_gb),
    'XGBoost': accuracy_score(y_test, y_pred_xgb)
}

plt.figure(figsize=(8, 5))
sns.barplot(x=list(models.keys()), y=list(models.values()), palette="viridis")
plt.title("Ensemble Model Comparison – Sentiment Analysis")
plt.ylabel("Accuracy")
plt.xticks(rotation=15)
plt.show()

def predict_sentiment(review):
    cleaned = clean_text(review)
    vector = tfidf.transform([cleaned]).toarray()
    pred = xgb.predict(vector)
    return "Positive 😊" if pred[0] == 1 else "Negative 😞"

# Example
print(predict_sentiment("This movie was absolutely fantastic!"))

