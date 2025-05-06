import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
file_path = 'customer_shopping_data.csv'  # Ruta del archivo
df = pd.read_csv(file_path)

# Seleccionamos las columnas que vamos a usar para el clustering
# Usamos 'price' y 'quantity' como las características para el clustering
X = df[['price', 'quantity']]

# Crear el modelo KMeans con 3 clusters (por ejemplo, puedes cambiar el número de clusters según sea necesario)
kmeans = KMeans(n_clusters=3, random_state=42)

# Entrenar el modelo KMeans
kmeans.fit(X)

# Predecir los clusters
y_pred = kmeans.predict(X)

# Agregar los resultados del clustering al DataFrame
df['Cluster'] = y_pred

# Calcular el Silhouette Score para evaluar el rendimiento del clustering
silhouette_avg = silhouette_score(X, y_pred)

# Mostrar el Silhouette Score
print("Silhouette Score:", silhouette_avg)

# Mostrar los centros de los clusters
print("\nCentros de los clusters:\n", kmeans.cluster_centers_)

# Mostrar los primeros 10 datos con sus clusters asignados
print("\nPrimeros 10 datos con clusters asignados:")
print(df[['price', 'quantity', 'Cluster']].head(10))

# Visualización del clustering
plt.figure(figsize=(8,6))
plt.scatter(df['price'], df['quantity'], c=df['Cluster'], cmap='viridis')
plt.title('Clustering de Productos (KMeans)')
plt.xlabel('Precio')
plt.ylabel('Cantidad')
plt.colorbar(label='Cluster')
plt.show()