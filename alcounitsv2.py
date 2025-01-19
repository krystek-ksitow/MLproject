import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

url1 = 'https://raw.githubusercontent.com/krystek-ksitow/MLproject/refs/heads/main/data/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'

cl_t_imp = ['Standard_Alcoholunits_Last_28days']
dta = pd.read_csv(url1, usecols=cl_t_imp)

dta['Standard_Alcoholunits_Last_28days'] = pd.to_numeric(dta['Standard_Alcoholunits_Last_28days'], errors='coerce')
dta_clean = dta.dropna(subset=['Standard_Alcoholunits_Last_28days'])
dta_sort = dta_clean.sort_values(by='Standard_Alcoholunits_Last_28days', ascending=True)

#original elbow (found in alcounits.py) said that 3 clusters are the most optimal, however I have decided that a sepparate cluster for people who consumed 0 alcounits is in order, therefore I shall sepparate it

z_cons = dta_sort[dta_sort['Standard_Alcoholunits_Last_28days'] == 0]
nz_cons = dta_sort[dta_sort['Standard_Alcoholunits_Last_28days'] > 0]

scaler = StandardScaler()
nz_cons_dta_scal = scaler.fit_transform(nz_cons)

kmeans = KMeans(n_clusters=3)
nz_cons = nz_cons.copy() #we create a copy as python was screaming about SettingWithCopyWarning
nz_cons['Clusternumber'] = kmeans.fit_predict(nz_cons)
z_cons = z_cons.copy() #copy again, same reason
z_cons['Clusternumber'] = 10

fin_dta = pd.concat([z_cons, nz_cons]).sort_index() #concatinate the 2 into 1

#Plot
colors = ['red', 'blue', 'green', 'orange']
for cluster in fin_dta['Clusternumber'].unique():
    cluster_data = fin_dta[fin_dta['Clusternumber'] == cluster]
    plt.scatter(
        cluster_data['Standard_Alcoholunits_Last_28days'], 
        [cluster] * len(cluster_data),
    )

plt.show()
