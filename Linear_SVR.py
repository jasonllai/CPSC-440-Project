import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

def train_and_predict(X_train, X_test, y_train, y_test):
    predictions = np.zeros_like(y_test)
    for i in range(y_train.shape[1]):
        model = make_pipeline(StandardScaler(), LinearSVR(random_state=42, max_iter=10000, tol=1e-5, C=1.0))
        model.fit(X_train, y_train[:, i])
        predictions[:, i] = model.predict(X_test)
    return predictions

def evaluate_model(predictions, y_test):
    mrrmse_score = np.sqrt(np.square(y_test - predictions).mean(axis=1))
    print(f'RMMSE = {mrrmse_score.mean()}')

def main():
    de_train, id_map = load_data()
    df, num_df, output_names = preprocess_data(de_train, id_map)
    
    test_size = min(255, len(num_df)) / len(num_df)
    
    X_train, X_test, y_train, y_test = train_test_split(num_df.iloc[:, :2], de_train.iloc[:, 5:], test_size=test_size, shuffle=True)
    predictions = train_and_predict(X_train, X_test, y_train.values, y_test.values)
    evaluate_model(predictions, y_test.values)

    output = pd.DataFrame(predictions, index=id_map.index, columns=output_names)

    output.to_csv('./files/SVR_submission.csv')

if __name__ == '__main__':
    main()
