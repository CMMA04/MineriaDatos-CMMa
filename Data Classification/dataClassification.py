import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('customer_shopping_data.csv')

# Para este ejemplo, seleccionaremos algunas columnas para la clasificación.
# Seguiremos usando cantidad y precio 
# Necesitamos transformar 'quantity' en una variable categórica para que KNN funcione correctamente.

# Vamos a convertir 'quantity' en una variable categórica (por ejemplo, bajo, medio, alto) usando cortes en los valores.
df['quantity_class'] = pd.cut(df['quantity'], bins=[0, 20, 40, 60, 80, 100], labels=['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto', 'Muy Alto'])

# Usamos 'price' como la variable independiente
X = df[['price']]

# Convertimos la variable dependiente (cantidad clasificada) a números para usarla en la clasificación
le = LabelEncoder()
y = le.fit_transform(df['quantity_class'])

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo KNN con 5 vecinos
knn = KNeighborsClassifier(n_neighbors=5)

# Entrenaremos el modelo y realizaremos las precidcciones
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


print("Accuracy del modelo KNN:", accuracy)

# Predicciones
predictions = le.inverse_transform(y_pred[:10])  # Convertir predicciones numéricas de vuelta a etiquetas
print("Predicciones para las primeras 10 instancias:", predictions)

#Tiene sentido que salga bajo por que como se realizo en el test anterior no tiene una relacion fuerte la cantidad de productos con el precio