import os
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

copecolumns = dta.columns[dta.columns.str.startswith('COPE')]
datasets = {}

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
        epochs = 100
        learning_rate = 0.01
        weights = np.zeros(x_train.shape[1])
        bias = 0
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            predictions = np.dot(x_train, weights) + bias
            residuals = predictions - y_train
            train_loss = np.mean(residuals*2)
            train_losses.append(train_loss)

            val_predictions = np.dot(x_test, weights) + bias
            val_residuals = val_predictions - y_test
            val_loss = np.mean(val_residuals**2)
            val_losses.append(val_loss)

            gradient_w = 2 * np.dot(x_train.T, residuals) / len(y_train)
            gradient_b = 2 * np.mean(residuals)
            weights -= learning_rate * gradient_w
            bias -= learning_rate * gradient_b

        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"Results for {cope_col}:")
        print(f"  R^2: {r2:.3f}")
        print(f"  Mean Squared Error: {mse:.3f}")
        print(f"  Mean Absolute Error: {mae:.3f}")
        print("-------------------------------")

        plt.scatter(x_test, y_test, label="Actual", color="red", alpha=0.7)
        plt.scatter(x_test, y_pred, label="Predicted", color="green", alpha=0.7)
        plt.xlabel("Alcohol Units Consumed (Last 28 Days)")
        plt.ylabel("Coping Mechanism Score")
        plt.title(f"Regression Results for {cope_col}")
        plt.legend()

        regrs_plotpath = os.path.join(output_dir, f"COPE{cope_col}_regression.png")
        plt.savefig(regrs_plotpath)

        plt.show()

        plt.subplot(1, 2, 2)
        plt.plot(range(epochs), train_losses, label="Training Loss", color="blue")
        plt.plot(range(epochs), val_losses, label="Validation Loss", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.title(f"Loss Curves for {cope_col}")
        plt.legend()
        plt.tight_layout()

        lossplot_path = os.path.join(output_dir, f"COPE{cope_col}_loss.png")
        plt.savefig(lossplot_path)

        plt.show()

    except Exception as e:
        print(f"Error with {cope_col}: {e}")
