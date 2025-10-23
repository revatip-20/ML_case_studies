# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load the dataset
df = pd.read_csv("advertising.csv")

# Display basic info
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Convert timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Extract useful time-based features
df['Hour'] = df['Timestamp'].dt.hour
df['Month'] = df['Timestamp'].dt.month

# Drop unnecessary columns
df = df.drop(['Ad Topic Line', 'City', 'Country', 'Timestamp'], axis=1)

# Features (X) and target (y)
X = df.drop('Clicked on Ad', axis=1)
y = df['Clicked on Ad']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Display feature importance (coefficients)
importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})
importance = importance.sort_values(by='Coefficient', ascending=False)
print(importance)

# Visualize
plt.figure(figsize=(8,5))
sns.barplot(x='Coefficient', y='Feature', data=importance)
plt.title("Feature Importance (Logistic Regression)")
plt.show()

# Example: Predict if a user clicks an ad
sample = np.array([[80, 35, 60000, 180, 1, 14, 5]])  # Example input
sample_scaled = scaler.transform(sample)
predicted_class = model.predict(sample_scaled)
print("Predicted Click (1=Yes, 0=No):", predicted_class[0])

