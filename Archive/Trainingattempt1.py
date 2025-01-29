from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def gettrainingdata(csv_url, train_size=0.8):
    import pandas as pd
    import numpy as np
    if not (0 < train_size < 1):
        raise ValueError("train_size must be between 0 and 1")

    dta = pd.read_csv(csv_url)
    fturs = dta.iloc[:, :-1].values
    labls = dta.iloc[:, -1].values
    indces = np.arange(len(fturs))
    np.random.shuffle(indces)
    split_pint = int(len(fturs) * train_size)

    x_train = fturs[indces[:split_pint]]
    x_test = fturs[indces[split_pint:]]
    y_train = labls[indces[:split_pint]]
    y_test = labls[indces[split_pint:]]

    return x_train, x_test, y_train, y_test

url = 'https://raw.githubusercontent.com/krystek-ksitow/MLproject/refs/heads/DSWBranch/data/DataWhole.csv'
dta = pd.read_csv(url)

copecolumns = [col for col in dta.columns if col.startswith("COPE")]

for cope_col in copecolumns:
    coping_table = dta[["Patient_ID", "Standard_Alcoholunits_Last_28days", cope_col]].copy()

    file_name = f"{cope_col}_table.csv"
    coping_table.to_csv(file_name, index=False)
    #print(f"saved as {file_name}")

cope_csvs = ['COPE_SelfDistraction_table.csv','COPE_UseOfEmotionalSupport_table.csv','COPE_BehavioralDisengagement_table.csv','COPE_positiveReframing_table.csv','COPE_Humor_table.csv','COPE_UseOfInstrumentalSupport_table.csv','COPE_Venting_table.csv','COPE_Planning_table.csv','COPE_Acceptance_table.csv','COPE_SelfBlame_table.csv','COPE_Religion_table.csv','COPE_Denial_table.csv','COPE_activeCoping_table.csv']
datasets = {}
for csv in cope_csvs:
    try:
        x_train, x_test, y_train, y_test = gettrainingdata(csv, train_size=0.8)
        datasets[csv] = {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
        }
      
        print(f"Sucess for: {csv} :D")
    except Exception as e:
        print(f"Error for {csv} :( : {e}")
regress_results = {}
for csv, data in datasets.items():
    try:
        x_train, x_test = data["x_train"], data["x_test"]
        y_train, y_test = data["y_train"], data["y_test"]
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        regress_results[csv] = {
            "model": model,
            "y_pred": y_pred,
        }

        print(f"Completed for: {csv} :D")
    except Exception as e:
        print(f"Error with {csv}: {e}")

y_test = datasets[csv]["y_test"]
y_pred = regress_results[csv]["y_pred"]

plt.scatter(range(len(y_test)), y_test, label="Actual", color="blue")
plt.scatter(range(len(y_pred)), y_pred, label="Predicted", color="red")
plt.xlabel("Samples")
plt.ylabel("Coping Mechanism Score")
plt.title(f"Regression Results for {csv}")
plt.legend()
plt.show()
