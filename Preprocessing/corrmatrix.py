import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url1 = 'https://raw.githubusercontent.com/krystek-ksitow/MLproject/refs/heads/DSWBranch/data/cleandatainonetable.csv'
dta = pd.read_csv(url1)
#cnames = dta.columns
#print(cnames)

colm_t_ex = ['Patient_ID','Standard_Alcoholunits_Last_28days']
num_dta = dta.drop(columns=colm_t_ex, errors='ignore')


corrmatrix = num_dta.corr()

sns.heatmap(corrmatrix, annot=True,cmap='coolwarm')
plt.show()
