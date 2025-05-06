import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt


file_path = 'customer_shopping_data.csv'  # Ruta del archivo
df = pd.read_csv(file_path)

#Utilizaremos la columna de categoria
text_data = df['category'].dropna()  

# Unir todos los textos en una sola cadena
text = " ".join(text_data.astype(str))

# Crear la nube de palabras
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Mostrar la nube de palabras
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Quitar los ejes
plt.title('Nube de Palabras')
plt.show()