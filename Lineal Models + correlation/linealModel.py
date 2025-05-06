import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


file_path = 'customer_shopping_data.csv'  
df = pd.read_csv(file_path)

# Vamos a crear un modelo lineal con 'quantity' como variable dependiente y 'price' como independiente
X = df[['price']]  # Variable independiente (precio)
y = df['quantity']  # Variable dependiente (cantidad)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo lineal
model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones sobre los datos de prueba
y_pred = model.predict(X_test)

# Calcular el R^2 (coeficiente de determinación)
r2 = r2_score(y_test, y_pred)

# Mostrar el R^2
print("R^2 score:", r2)

#Yo tomare que si el R^2 es menos al 70% entonces para mi no es suficientemente fuerte la relacion
if (r2 < .7) :
  print("Como el R^2 es menor a 70%, no es lo suficientemente fuerte la relacion por lo que la cantidad de productos comprados no esta relacionado con el precio ")
else:
  print("La cantiidad de productos comprados esta relacionado con su precio")



# Crear gráfico de dispersión de los datos de prueba
plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color='blue', label='Datos reales')

# Graficar la línea de regresión
plt.plot(X_test, y_pred, color='red', label='Línea de regresión')

plt.title('Modelo de Regresión Lineal: Precio vs Cantidad')
plt.xlabel('Precio')
plt.ylabel('Cantidad')
plt.legend()
plt.show()