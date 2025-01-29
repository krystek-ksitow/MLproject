#I heard that we are supposed to leave things that dont work anyways so that one can see our process

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


url1 = 'https://raw.githubusercontent.com/krystek-ksitow/MLproject/refs/heads/main/data/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'

cl_t_imp = ['Standard_Alcoholunits_Last_28days']
dta = pd.read_csv(url1, usecols = cl_t_imp)

#======================================================================================
#dta_sort = dta.sort_values(by='Standard_Alcoholunits_Last_28days', ascending=True)
#dta_clean = dta.dropna(subset=['Standard_Alcoholunits_Last_28days'])
#dta_sort = dta_clean.sort_values(by='Standard_Alcoholunits_Last_28days', ascending=True)
#print(dta_sort)
#======================================================================================
#It seems that the numbers are nor numbers but strings so 9.5 is recognized as the biggest number and not. This villany shall not stand

dta['Standard_Alcoholunits_Last_28days'] = pd.to_numeric(dta['Standard_Alcoholunits_Last_28days'], errors='coerce')
dta_clean = dta.dropna(subset=['Standard_Alcoholunits_Last_28days'])
dta_sort = dta_clean.sort_values(by='Standard_Alcoholunits_Last_28days', ascending=True)
#print(dta_sort)

scaler = StandardScaler()
dta_scal = scaler.fit_transform(dta_sort)
inertia = []
k_range = range(1, 7)  
for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(dta_scal)
    inertia.append(kmeans.inertia_)

plt.plot(k_range, inertia)
#plt.show()

#Elbow says 3 clusters is optimal, might still go for 4, for now we can do 3

kmeans = KMeans(n_clusters=3)
dta_sort['Clusternumber'] = kmeans.fit_predict(dta_sort)
#print(dta_sort)

#======================================================================================
#for cluster in dta_sort['Clusternumber'].unique():
    #cluster_data = dta_sort[dta_sort['Clusternumber'] == cluster]
    #plt.scatter(cluster_data.index, cluster_data['Standard_Alcoholunits_Last_28days'], label=f'Cluster {cluster}')


#plt.show()
#======================================================================================
#...what?

colors = ['red', 'blue', 'green', 'purple']

for cluster in dta_sort['Clusternumber'].unique():
    cluster_dta = dta_sort[dta_sort['Clusternumber'] == cluster]
    plt.scatter(
        cluster_dta['Standard_Alcoholunits_Last_28days'], 
        [cluster] * len(cluster_dta),
        color=colors[cluster], 
    )

plt.show()

#I guess thats ok
