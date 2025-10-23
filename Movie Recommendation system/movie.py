import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Example dataset
data = {
    'Movie': ['Inception', 'Avatar', 'Titanic', 'Avengers', 'Interstellar',
               'Joker', 'Iron Man', 'The Dark Knight', 'Toy Story', 'Frozen'],
    'Action': [1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
    'Adventure': [1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    'Romance': [0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
    'Sci-Fi': [1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
    'Drama': [0, 0, 1, 0, 1, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)
print(df)

# Features for clustering (excluding movie name)
X = df.drop('Movie', axis=1)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []  # Within Cluster Sum of Squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.show()

# Let's assume the optimal K is 3
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print(df)

# Reduce dimensions for visualization
pca = PCA(2)
pca_features = pca.fit_transform(X_scaled)

df['x'] = pca_features[:, 0]
df['y'] = pca_features[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(x='x', y='y', hue='Cluster', data=df, palette='Set1', s=100)
for i in range(df.shape[0]):
    plt.text(df['x'][i]+0.02, df['y'][i], df['Movie'][i], fontsize=9)
plt.title("Movie Clusters (K-Means)")
plt.show()

def recommend_movies(movie_name):
    movie_name = movie_name.lower()
    if movie_name not in df['Movie'].str.lower().values:
        return "❌ Movie not found in database."
    
    cluster = df[df['Movie'].str.lower() == movie_name]['Cluster'].values[0]
    recommendations = df[df['Cluster'] == cluster]['Movie'].values
    print(f"\n🎬 Movies similar to '{movie_name.title()}':")
    for rec in recommendations:
        if rec.lower() != movie_name:
            print("-", rec)

# Example
recommend_movies("Inception")

