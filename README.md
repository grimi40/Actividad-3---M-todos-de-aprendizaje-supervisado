import pandas as pd
from sklearn.cluster import KMeans

# Cargar datos
datos = pd.read_csv("datos_transporte_masivo.csv")

# Seleccionar características
caracteristicas = ["pasajeros", "distancia", "tiempo"]

# Normalizar datos
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
datos_normalizados = scaler.fit_transform(datos[caracteristicas])

# Crear modelo KMeans
kmeans = KMeans(n_clusters=4)

# Ajustar el modelo a los datos
kmeans.fit(datos_normalizados)

# Obtener etiquetas de clúster
etiquetas_cluster = kmeans.labels_

# Agregar etiquetas de clúster a datos originales
datos["clúster"] = etiquetas_cluster

# Analizar resultados
print(datos.groupby("clúster").mean())

