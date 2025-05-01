import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


archivo = 'customer_shopping_data.csv'
datos = pd.read_csv(archivo)

sns.set_palette("dark")


columns = ['age', 'quantity', 'price']

plt.figure(figsize=(8, 6))


plt.subplot(2, 3, 1)
datos['payment_method'].value_counts().plot(kind='pie',autopct='%1.1f%%', startangle=2000)
plt.title('Distribucion de tipos de pago ')

plt.subplot(2, 3, 2)
datos['quantity'].plot(kind='hist', bins=20, color='c', edgecolor='black')
plt.title('Distribución de la Cantidad')
plt.xlabel('Cantidad')

plt.subplot(2, 3, 3)
sns.boxplot(x=datos['price'])
plt.title('Distribución de Precios')


plt.subplot(2, 3, 4)
sns.scatterplot(x='quantity', y='price', data=datos, color='g')
plt.title('Relación entre Cantidad y Precio')


plt.subplot(2, 3, 5)
sns.lineplot(x='age', y='price', data=datos, marker='o', color='r')
plt.title('Precio en función de la Edad')
 

plt.tight_layout()
plt.show()