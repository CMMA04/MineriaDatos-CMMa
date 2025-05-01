import pandas as pd
import scipy.stats as stats

# Cargar el dataset
file_path = 'customer_shopping_data.csv'
df = pd.read_csv(file_path)

# Ejemplo: comparar precios entre grupos de edad (puedes ajustar las columnas de acuerdo a tus datos)
# Agrupar los datos en grupos de edades (por ejemplo, 18-30, 31-40, 41-50, etc.)
bins = [18, 30, 40, 50, 60, 70]
labels = ['18-30', '31-40', '41-50', '51-60', '61-70']
df['rango'] = pd.cut(df['age'], bins=bins, labels=labels)

# Realizar ANOVA para comparar precios entre diferentes grupos de edad
anova_result = stats.f_oneway(
    df[df['rango'] == '18-30']['price'],
    df[df['rango'] == '31-40']['price'],
    df[df['rango'] == '41-50']['price'],
    df[df['rango'] == '51-60']['price'],
    df[df['rango'] == '61-70']['price']
)

print("ANOVA Result: F-statistic = ", anova_result.statistic, "p-value = ", anova_result.pvalue)

# Comparar dos grupos (por ejemplo, edad menor de 40 años y mayor de 40 años)
group_1 = df[df['age'] < 40]['price']
group_2 = df[df['age'] >= 40]['price']

# Realizar el T-test
t_stat, p_value = stats.ttest_ind(group_1, group_2)

print("T-test Result: t-statistic = ", t_stat, "p-value = ", p_value)