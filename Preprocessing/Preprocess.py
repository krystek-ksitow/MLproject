import pandas as pd

url1 = 'https://raw.githubusercontent.com/krystek-ksitow/MLproject/refs/heads/krist/data/Coping%20Orientations%20to%20Problems.csv'
url2 = 'https://raw.githubusercontent.com/krystek-ksitow/MLproject/refs/heads/main/data/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'

dta1 = pd.read_csv(url1)
dta2 = pd.read_csv(url2)

dta = pd.merge(dta1,dta2)
#dta_colnames = dta.columns
#print(dta_colnames)
col_to_excl = ['Gender_ 1=female_2=male', 'Age',
       'Handedness', 'Education', 'DRUG', 'DRUG_0=negative_1=Positive',
       'Unnamed: 7', 'Smoking',
       'Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)',
       'SKID_Diagnoses', 'SKID_Diagnoses 1', 'SKID_Diagnoses 2',
       'Comments_SKID_assessment', 'Hamilton_Scale', 'BSL23_sumscore',
       'BSL23_behavior', 'AUDIT',
       'Alcohol_Dependence_In_1st-3rd_Degree_relative', 'Relationship_Status']

dta_clean = dta.drop(columns=col_to_excl, errors='ignore')

dta_clean['Unnamed: 0'] = dta_clean['Unnamed: 0'].str.replace('sub-', '', regex=False)
dta_clean['Unnamed: 0'] = dta_clean['Unnamed: 0'].astype(int)
dta_clean.rename(columns={'Unnamed: 0': 'Patient_ID'}, inplace=True)
dta_clean = dta_clean.dropna(subset=['Standard_Alcoholunits_Last_28days'])

#It would appear that some vile wretch put ',' as opposed to '.' in a few cells in 'Standard_Alcoholunits_Last_28days', which prompts me to hit the column with:
dta_clean['Standard_Alcoholunits_Last_28days'] = dta_clean['Standard_Alcoholunits_Last_28days'].str.replace(',','.', regex=False)
dta_clean['Standard_Alcoholunits_Last_28days'] = dta_clean['Standard_Alcoholunits_Last_28days'].astype(float)

#path = r'[Censored]'
#dta_clean.to_csv(path, index=False)
