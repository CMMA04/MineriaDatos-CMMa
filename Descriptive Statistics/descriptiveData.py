import pandas as pd

archivo = 'customer_shopping_data.csv' 
datos = pd.read_csv(archivo)
descriptive_data = datos.describe()
print(descriptive_data)