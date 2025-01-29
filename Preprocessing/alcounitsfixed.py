import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

url1 = 'https://raw.githubusercontent.com/krystek-ksitow/MLproject/refs/heads/DSWBranch/data/cleandatainonetable.csv'
cl_t_imp = ['Patient_ID','Standard_Alcoholunits_Last_28days']
dta = pd.read_csv(url1, usecols = cl_t_imp)

scaler = StandardScaler()
dta_scal = scaler.fit_transform(dta)
inertia = []
k_range = range(1, 7)  
for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(dta_scal)
    inertia.append(kmeans.inertia_)

#plt.plot(k_range, inertia)
#plt.show()

#Elbow says 3 is the optimal number of clusters, we are adding a 4th one for people who drank 0 alcounits

z_cons = dta[dta['Standard_Alcoholunits_Last_28days'] == 0].copy()
nz_cons = dta[dta['Standard_Alcoholunits_Last_28days'] > 0].copy()

scaler = StandardScaler()
nz_cons_scal = scaler.fit_transform(nz_cons)

kmeans = KMeans(n_clusters=3)
nz_cons['Clusternumber'] = kmeans.fit_predict(nz_cons_scal)
z_cons['Clusternumber'] = 10

dta_concat = pd.concat([z_cons, nz_cons]).reset_index(drop=True)
#Plot
colors = ['red', 'blue', 'green', 'orange']
for cluster in dta_concat['Clusternumber'].unique():
    cluster_data = dta_concat[dta_concat['Clusternumber'] == cluster]
    plt.scatter(
        cluster_data['Standard_Alcoholunits_Last_28days'], 
        [cluster] * len(cluster_data),
    )

#plt.show()

path1 = r'D:\The D\python projects\machinelearning\GODHELPME.csv'
dta_concat.to_csv(path1, index=False)

#we also concatinate that with original table to have everything in one csv file

dtawhole = pd.read_csv(url1)

dta_fin = pd.merge(dta_concat,dtawhole, on='Patient_ID', how = "outer")
#coltm = dta_fin.pop('Patient_ID')
#dta_fin.insert(0, 'Patient_ID', coltm) #moved so that patientid column is the first one

path2 = r'D:\The D\python projects\machinelearning\dtamergednotfinal.csv'
dta_fin.to_csv(path2, index=False)

print("z_cons shape:", z_cons.shape)
print("nz_cons shape:", nz_cons.shape)
print("dta_concat shape:", dta_concat.shape)
print("dta_fin shape:", dta_fin.shape)
