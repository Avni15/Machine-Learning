#importing libraries
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

Data = {'x': [25,34,22,27,33,33,31,22,35,34,67,54,57,43,50,57,59,52,65,47,49,48,35,33,44,45,38,43,51,46],
        'y': [79,51,53,78,59,74,73,57,69,75,51,32,40,47,53,36,35,58,59,50,25,20,14,12,20,5,29,27,8,7]
        }

#converting data into dataframe using pandas
df = pd.DataFrame(Data,columns=['x','y'])

#creating kmeans cluster model and fitting the data into the mdoel
kmeans = KMeans(n_clusters=3).fit(df)

#plotting data points into scatter plot and differentiating the clusters using different clusters
plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), alpha=0.5)

#plotting the final graph
plt.show()