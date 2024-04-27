import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

def load_data():
    de_train = pd.read_parquet('./data/de_train.parquet')
    id_map = pd.read_csv('./data/id_map.csv', index_col=0)
    return de_train, id_map

def preprocess_data(de_train, id_map):
    encoder = preprocessing.LabelEncoder()
    df = pd.DataFrame({
        'cell_type': encoder.fit_transform(id_map['cell_type']),
        'sm_name': encoder.fit_transform(id_map['sm_name'])
    })
    num_df = pd.DataFrame({
        'cell_type': encoder.fit_transform(de_train['cell_type']),
        'sm_name': encoder.fit_transform(de_train['sm_name']),
        'sm_lincs_id': encoder.fit_transform(de_train['sm_lincs_id']),
        'smiles': encoder.fit_transform(de_train['SMILES']),
        'control': de_train['control'].fillna(False).astype(int)
    })
    output_names = de_train.iloc[:, 5:].columns.values.tolist()
    return df, num_df, output_names

def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mrrmse_score = np.sqrt(np.square(y_test - predictions).mean(axis=1))
    print(f'RMMSE = {mrrmse_score.mean()}')
    plt.scatter(range(len(mrrmse_score)), mrrmse_score)
    plt.xlabel('Iterations')
    plt.ylabel('MRRMSE')
    plt.savefig('./files/DT_MRRMSE.png')
    return predictions

def main():
    de_train, id_map = load_data()
    df, num_df, output_names = preprocess_data(de_train, id_map)
    X_train, X_test, y_train, y_test = train_test_split(num_df.iloc[:, :2], de_train.iloc[:, 5:], test_size=0.3, shuffle=True)
    model = train_decision_tree(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    model.fit(num_df.iloc[:, :2], de_train.iloc[:, 5:])
    predictions = model.predict(df)
    output = pd.DataFrame(predictions, index=id_map.index, columns=output_names)
    output.to_csv('./files/DT_submission.csv')

if __name__ == '__main__':
    main()
