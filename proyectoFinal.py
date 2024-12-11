import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

def clean_numeric_value(value):
    try:
        if isinstance(value, str):
            return float(''.join(filter(str.isdigit, value)))
        return float(value)
    except:
        return None

# Cargar datos
dataset = pd.read_excel("InfluenciaDatos.xlsx")

# Limpiar columnas
columns_to_drop = ['Marca temporal']
dataset_cleaned = dataset.drop(columns=columns_to_drop)
dataset_cleaned.columns = dataset_cleaned.columns.str.strip()

# Columnas numéricas
numeric_columns = [
    '¿En qué nivel crees que los "me gusta" o comentarios positivos impactan en cómo te sientes contigo mismo(a)?',
    '¿Con qué frecuencia publicas contenido relacionado con tu apariencia personal (fotos, videos)?'
]
# Columnas de categorías 
categorical_columns = [
    '¿Cuántas horas al día pasas en redes sociales?',
    '¿Con qué frecuencia comparas tu apariencia con la de otros usuarios en redes sociales?',
]
# Columnas de texto
text_columns = [
    '¿Qué plataforma sientes que más afectan tu percepción personal?',
    '¿Qué tipo de contenido en redes sociales sientes que mejora tu autoestima y cuál disminuye tu confianza?',
    '¿Consideras que las redes sociales han influido en cómo defines la belleza o el éxito personal? Explica cómo.',
    '¿Has sentido alguna vez presión por mostrar una imagen perfecta o idealizada en tus publicaciones? Si es así, ¿cómo te afecta emocionalmente?',
    '¿Qué estrategias usas o conoces para evitar que las redes sociales afecten negativamente tu autoestima?',
    'Describe una experiencia en redes sociales que haya afectado positivamente tu autoimagen.'
]

# Limpiar datos numéricos
for column in numeric_columns:
    dataset_cleaned[column] = dataset_cleaned[column].apply(clean_numeric_value)

data_numeric = dataset_cleaned[numeric_columns]
data_categorical = dataset_cleaned[categorical_columns]
data_text = dataset_cleaned[text_columns]

# Eliminar filas con valores nulos en columnas numéricas
data_numeric = data_numeric.dropna()

print("\nEstadísticas descriptivas después de limpieza:")
print(data_numeric.describe())

# Visualizaciones
for column in numeric_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data_numeric[column].dropna(), kde=True)
    plt.title(f"Distribución de {column}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Nube de palabras para columnas de texto
for column in text_columns:
    text_data = " ".join(dataset_cleaned[column].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud)
    plt.title(f"Nube de palabras: {column}")
    plt.axis("off")
    plt.show()

# Clustering
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# Método del codo
wcss = []
K = range(2, 8)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K, wcss, 'bo-')
plt.title("Método del Codo")
plt.xlabel("Número de Clústeres")
plt.ylabel("WCSS")
plt.show()

# Aplicar K-means
kmeans = KMeans(n_clusters=3, random_state=42)
data_numeric['Cluster'] = kmeans.fit_predict(data_scaled)

# Visualizar clusters
plt.figure(figsize=(10, 6))
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=data_numeric['Cluster'])
plt.title("Clusters")
plt.xlabel(numeric_columns[0])
plt.ylabel(numeric_columns[1])
plt.show()

# Clasificación
satisfaction_mapping = {
    "Nunca": 0,
    "A veces": 1,
    "Frecuentemente": 1,
    "Siempre": 2,
}

dataset_cleaned['Etiqueta'] = dataset_cleaned['¿Con qué frecuencia comparas tu apariencia con la de otros usuarios en redes sociales?'].map(satisfaction_mapping)

# Usar filas con datos numéricos completos 
valid_indices = data_numeric.index
X = data_numeric.loc[valid_indices].drop(columns=['Cluster'])
y = dataset_cleaned.loc[valid_indices, 'Etiqueta']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title("Matriz de Confusión")
plt.show()