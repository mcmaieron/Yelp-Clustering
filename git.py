# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 19:49:27 2019

@author: mcmai
"""

import json
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import geopandas as gpd
from nltk.stem import SnowballStemmer
from collections import OrderedDict
from scipy.cluster.hierarchy import fcluster,dendrogram, linkage, cophenet
from sklearn import mixture
from scipy.spatial.distance import pdist


def extract(obj):
    ob = json.loads(obj)
    for key, value in list(ob.items()):
        if(key == 'categories'):
            if (['Restaurants','Food'] not in value):
                del ob[key]
            else:
                ob[key] = ','.join(value)    
    return ob


def process(data):
    return " ".join([SnowballStemmer('english').stem(word) for word in data])

def convert(j):
    for i in t15:
        if(i == j):
            return(t15.index(i)+1)

def appending(j):
    if j in t15:
        c.append(j)
        
def evaluate_clusters(final_df,max_clusters):
    error = np.zeros(max_clusters+1)
    error[0] = 0;
    for k in range(1,max_clusters+1):
        kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
        kmeans.fit_predict(final_df)
        error[k] = kmeans.inertia_
    plt.figure(1)
    plt.plot(range(1,len(error)),error[1:])
    plt.xlabel('Number of clusters')
    plt.ylabel('Error')
        
df = pd.read_json(r'F:\Mestrado\Computacao\KDD\TrabalhoFinal\dados\business.json', lines=True)



final_df = df[['latitude','longitude','city','categories']]
final_df = final_df.loc[final_df['city'] == 'Toronto']
final_df = final_df[['latitude','longitude','categories']]
final_df = final_df.dropna(subset=['categories'])
final_df[['latitude','longitude']] = (final_df[['latitude','longitude']].sub(final_df[['latitude','longitude']].mean())).divide(final_df[['latitude','longitude']].std())
top15_df = final_df.copy()
cat_list = final_df['categories'].tolist()
x = []
for i in cat_list:
     x += [i.split(',')]

        
count={}
for i in x:
    for j in i:
        if j not in count:
            count[j]=1
        else:
            count[j] = count[j] +1
top15_names=[]
od = OrderedDict(sorted(count.items(), key=lambda kv:kv[1], reverse=True))    
t15 = list(od)[1:16] # Eliminating Restaurants
top15_names = t15
t15_counts = list(od.values())[1:16]
trimmed_list = []
unwanted = []
c =[]
for i in final_df['categories']:
    row = i.split(',')
    c += [[convert(j) for j in row if j in t15]]
    #c += [[ j for j in row if j in t15]]
    
    #trimmed_list.append(row)
#print(c)
top15_df['categories'] = c
top15_df = top15_df.reset_index(drop = True)
a = np.zeros(shape=(len(top15_df['categories']),15))
i=0
for j in c:
    for k in j:
        a[i,(k-1)] = 1
    i+=1
t15= pd.DataFrame.from_records(a)
t15[['latitude','longitude']] = top15_df[['latitude','longitude']]
evaluate_clusters(t15,10) #evaluates error versus number of clusters




#K-means 
kmeans = KMeans(init='k-means++', n_clusters=5, n_init=100) 
km = kmeans.fit_predict(t15)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_


def labels(cluster):
    label_count ={}
    for i in cluster[0]:
        for j in top15_df['categories'][i]:
            if j not in label_count:
                label_count[j] = 1
            else:
                label_count[j] +=1
    ord_dict = OrderedDict(sorted(label_count.items(), key=lambda kv:kv[1], reverse=True))
    return ord_dict

def cluster_label(lc,t15_counts):
    maxi = 0
    for key,value in lc.items():
        temp = value/t15_counts[key-1]
        if(temp>maxi):
            maxi = temp
            index = key-1
    return top15_names[index]

all_labels = []
km_0 = np.where(km==0)
lc = labels(km_0)
all_labels.append(cluster_label(lc, t15_counts))

km_1 = np.where(km==1)
lc = labels(km_1)
all_labels.append(cluster_label(lc, t15_counts))

km_2 = np.where(km==2)
lc = labels(km_2)
all_labels.append(cluster_label(lc, t15_counts))

km_3 = np.where(km==3)
lc = labels(km_3)
all_labels.append(cluster_label(lc, t15_counts))

km_4 = np.where(km==4)
lc = labels(km_4)
all_labels.append(cluster_label(lc, t15_counts))




#Hierarchical
hc = linkage(t15, method ='complete',metric = 'euclidean')
cluster = fcluster(hc,5,criterion='distance')
all_labels_hc=[]

h_1 = np.where(cluster==1)
h= labels(h_1)
all_labels_hc.append(cluster_label(h, t15_counts))

h_2 = np.where(cluster==2)
h= labels(h_2)
all_labels_hc.append(cluster_label(h, t15_counts))

h_3 = np.where(cluster==3)
h= labels(h_3)
all_labels_hc.append(cluster_label(h, t15_counts))

h_4 = np.where(cluster==4)
h= labels(h_4)
all_labels_hc.append(cluster_label(h, t15_counts))

h_5 = np.where(cluster==5)
h= labels(h_5)
all_labels_hc.append(cluster_label(h, t15_counts))








#GMM
all_labels_gm=[]
gmm = mixture.GaussianMixture(n_components=5, covariance_type='spherical')
gmm.fit(t15)
gaussian = gmm.predict(t15)
mean = gmm.means_
covar = gmm.covariances_

g_0 = np.where(gaussian==0)
x= labels(g_0)
all_labels_gm.append(cluster_label(x, t15_counts))

g_1 = np.where(gaussian==1)
x= labels(g_1)
all_labels_gm.append(cluster_label(x, t15_counts))

g_2 = np.where(gaussian==2)
x= labels(g_2)
all_labels_gm.append(cluster_label(x, t15_counts))

g_3 = np.where(gaussian==3)
x= labels(g_3)
all_labels_gm.append(cluster_label(x, t15_counts))

g_4 = np.where(gaussian==4)
x= labels(g_4)
all_labels_gm.append(cluster_label(x, t15_counts))



#K-Means ploting
clusters = [km_0,km_1,km_2,km_3,km_4]
colors = ['red','green','yellow','magenta','cyan']
plt.figure(0)

x = 0
for i in clusters:
    for j in i:
        plt.scatter(t15['latitude'][j],t15['longitude'][j], color=colors[x])
    x=x+1

y = 0    
for i in centroids:
    plt.scatter(i[15],i[16],s=50,c='black',marker='*',linewidth=2)
    plt.annotate(all_labels[y],xy=(i[15],i[16]))
    y+=1
plt.title('K-means')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
legend = plt.legend([all_labels[i] for i in range(len(all_labels))],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=15)
#referenced Stack overflow for legend tag and annotate
plt.show()

#Hierarchical
clusters_hc = [h_5,h_1,h_2,h_3,h_4]
colors = ['red','green','yellow','magenta','cyan']
plt.figure(1)

x = 0
for i in clusters_hc:
    for j in i:
        plt.scatter(t15['latitude'][j],t15['longitude'][j], color=colors[x])
    x=x+1

plt.title('Hierarchical Clustering')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
legend = plt.legend([all_labels_hc[i] for i in range(len(all_labels_hc))],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=15)
#referenced Stack overflow for legend tag and annotate
plt.show()

#GMM
clusters_gmm = [g_0,g_1,g_2,g_3,g_4]
colors = ['red','green','yellow','magenta','cyan']
plt.figure(2)

x = 0
for i in clusters_gmm:
    for j in i:
        plt.scatter(t15['latitude'][j],t15['longitude'][j], color=colors[x])
    x=x+1

plt.title('GMM')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
legend = plt.legend([all_labels_gm[i] for i in range(len(all_labels_gm))],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=15)
#referenced Stack overflow for legend tag and annotate
plt.show()


##Ploting location## 
#loads US/Canada shapefile
world = gpd.read_file('F:\Mestrado\Computacao\KDD\TrabalhoFinal\dados\shape_mundo\World_Countries.shp')
toronto_boundaries = gpd.read_file(r'F:\Mestrado\Computacao\KDD\TrabalhoFinal\dados\toronto_boundaries\toronto_boundaries.shp')

#Set CRS as 4326 (WGS84)
crs = {'init': 'epsg:4326'}    

# Create a list with the pair of coordinate (LONG,LAT)    
geometry = [Point(xy) for xy in zip (cluster_df["longitude"],cluster_df["latitude"])]

# Create a datafreme with the column geometry and the crs
geo_df = gpd.GeoDataFrame(cluster_df,crs=crs, geometry=geometry)

#Sets the size of the graph
fig,ax = plt.subplots(figsize = (15,15))

#Plots the world shape
#world.plot(ax=ax, alpha=0.4, color="grey")
toronto_boundaries.plot(ax=ax, alpha=0.4, color="grey")
#Plots the points
geo_df.plot(ax=ax, markersize=20, column='cluster_id', cmap='Set1')
plt.title('Estimated number of clusters: %d Epsilon:%.3f Min Samples:%d')
plt.show()






