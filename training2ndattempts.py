from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def gettrainingdata(data, x_col, y_col, train_size=0.8):
    import pandas as pd
    import numpy as np
    if not (0 < train_size < 1):
        raise ValueError("train_size must be between 0 and 1")
    fturs = data[[x_col]].values
    labls = data[y_col].values
    indces = np.arange(len(fturs))
    np.random.shuffle(indces)
    split_point = int(len(fturs) * train_size)

    x_train = fturs[indces[:split_point]]
    x_test = fturs[indces[split_point:]]
    y_train = labls[indces[:split_point]]
    y_test = labls[indces[split_point:]]

    return x_train, x_test, y_train, y_test

url = 'https://raw.githubusercontent.com/krystek-ksitow/MLproject/refs/heads/DSWBranch/data/cleandatainonetable.csv'
dta = pd.read_csv(url)

copecolumns = dta.columns[dta.columns.str.startswith('COPE')]


datasets = {}
for cope_col in copecolumns:
    try:
        coping_table = dta[["Standard_Alcoholunits_Last_28days", cope_col]].dropna()
        x_train, x_test, y_train, y_test = gettrainingdata(coping_table, x_col="Standard_Alcoholunits_Last_28days", y_col=cope_col, train_size=0.82)
        datasets[cope_col] = {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
        }
        print(f"Success for: {cope_col} :D")
    except Exception as e:
        print(f"Error for {cope_col}: {e}")


for cope_col, data in datasets.items():
    try:
        x_train, x_test = data["x_train"], data["x_test"]
        y_train, y_test = data["y_train"], data["y_test"]

        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        plt.scatter(x_test, y_test, label="Actual", color="red", alpha=0.7)
        plt.scatter(x_test, y_pred, label="Predicted", color="green", alpha=0.7)
        plt.xlabel("Alcohol Units Consumed (Last 28 Days)")
        plt.ylabel("Coping Mechanism Score")
        plt.title(f"Regression Results for {cope_col}")
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Error with {cope_col}: {e}")
