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

#x_train, x_test, y_train, y_test = gettrainingdata('https://raw.githubusercontent.com/krystek-ksitow/MLproject/refs/heads/DSWBranch/data/cleandatainonetable.csv', train_size=0.8)
#print("Training Features:\n", x_train)
