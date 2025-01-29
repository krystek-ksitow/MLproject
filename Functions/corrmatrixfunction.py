def generate_corr_matrix(url, include_columns, output_file):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    try:
        dta = pd.read_csv(url)
        sel_dta = dta[include_columns]
        corr_matrix = sel_dta.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Corr_Matrix')
        plt.savefig(output_file, format='png')
        plt.show()
        plt.close()

        print(f"Correlation matrix saved as {output_file}")
    except KeyError as e:
        print(f"Some columns were not found in the data: {e}")

#generate_corr_matrix('https://raw.githubusercontent.com/krystek-ksitow/MLproject/refs/heads/DSWBranch/data/cleandatainonetable.csv', ['COPE_UseOfEmotionalSupport','COPE_BehavioralDisengagement','COPE_positiveReframing','COPE_Humor'], 'D:\The D\python projects\machinelearning\here.png')
