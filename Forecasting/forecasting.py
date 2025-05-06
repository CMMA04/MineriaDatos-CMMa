import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


file_path = 'customer_shopping_data.csv'
df = pd.read_csv(file_path)

# Verificar el tamaño del DataFrame
print(f"Tamaño del dataset: {len(df)} filas")


df = df.head(1000)  # Limitamos a 1000 filas por que si no explota

# Como el dataset no tiene fila de fechas, generamos una de ejemplo
df['date'] = pd.date_range(start='2025-01-01', periods=len(df), freq='D')

# Ordenar los datos por fecha (si no está ordenado)
df = df.sort_values('date')

# Usaremos 'quantity' como la variable dependiente (y) y 'date' como independiente (X).
# Convertir la fecha en formato numérico (para que sea compatible con la regresión)
df['date_numeric'] = np.arange(len(df))  # Usamos el índice temporal como la variable independiente

# Definir las variables independientes (X) y dependientes (y)
X = df[['date_numeric']]
y = df['quantity']

# Dividir los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Predecir para los próximos datos, en este caso predecuremos 10 fechas
future_dates = pd.date_range(start=df['date'].max(), periods=11, freq='D')[1:] 

# Convertir las futuras fechas a formato numérico
future_dates_numeric = np.arange(len(df), len(df) + 10).reshape(-1, 1)

# Predecir la cantidad para las fechas futuras
future_predictions = model.predict(future_dates_numeric)

# Mostrar las predicciones futuras
print("Predicciones para las próximas 10 fechas:", future_predictions)

plt.figure(figsize=(10, 6))
plt.plot(df['date'], y, label='Datos Originales')
plt.plot(df['date'].iloc[len(df)-len(X_test):], y_pred, color='red', label='Predicciones')
plt.scatter(future_dates, future_predictions, color='green', label='Predicciones Futuras')
plt.title('Predicción de la Serie Temporal usando Regresión Lineal')
plt.xlabel('Fecha')
plt.ylabel('Cantidad')
plt.legend()
plt.show()