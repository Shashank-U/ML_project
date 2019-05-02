import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
import random as rd

data=pd.read_csv('D:\shashu\Mini Project2\Mall_Customers.csv')
data.describe()
X = data.iloc[:, [3, 4]].values


K=5
m=200
Cluster_centroids_matrix=np.array([]).reshape(2,0)

#step 1
for i in range(K):
    random=rd.randint(0,m-1)
    Cluster_centroids_matrix=np.c_[Cluster_centroids_matrix,X[random]]
    

#step2
iterations=100
Output=defaultdict()
OutputDic={}
for n in range(iterations):
    #step 2.a
    Euclidian_distance_matrix=np.array([]).reshape(m,0)
    for k in range(K):
        temp_Dist=np.sum((X-Cluster_centroids_matrix[:,k])**2,axis=1)
        Euclidian_distance_matrix=np.c_[Euclidian_distance_matrix,temp_Dist]
        
    CIndex=np.argmin(Euclidian_distance_matrix,axis=1)+1
    #step 2.b
    Y={}
    for k in range(K):
        Y[k+1]=np.array([]).reshape(2,0)
    for i in range(m):
        Y[CIndex[i]]=np.c_[Y[CIndex[i]],X[i]]
     
    for k in range(K):
        Y[k+1]=Y[k+1].T
        
        
    for k in range(K):
        Cluster_centroids_matrix[:,k]=np.mean(Y[k+1],axis=0)
        
    OutputDic=Y

plt.scatter(X[:,0],X[:,1],c='black',label='unclustered data')
plt.xlabel('Income')
plt.ylabel('Number of transactions')
plt.legend()
plt.title('Plot of data points')
plt.show()

color=['green','red','blue','magenta','yellow']
labels=['cluster1','cluster2','cluster3','cluster4','cluster5']
for k in range(K):
    plt.scatter(OutputDic[k+1][:,0],OutputDic[k+1][:,1],c=color[k],label=labels[k])
plt.scatter(Cluster_centroids_matrix[0,:],Cluster_centroids_matrix[1,:],s=100,c='cyan',label='Centroids')
plt.xlabel('Income')
plt.ylabel('Number of transactions')
plt.legend()
plt.show()