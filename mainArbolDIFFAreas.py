import matplotlib.pyplot as plt  
import numpy as np  
from sklearn.cluster import KMeans  
import pandas as pd

datos = pd.read_csv('GenelbaFichadasArbolesDIFFArea.csv')
DF = pd.DataFrame(datos)
X = np.array(DF)


kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
print(kmeans.cluster_centers_)  
print(kmeans.labels_)
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')  
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')  

plt.show()

