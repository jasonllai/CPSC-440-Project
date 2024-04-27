import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

def load_data():
    de_train = pd.read_parquet('./data/de_train.parquet')
    id_map = pd.read_csv('./data/id_map.csv', index_col=0)
    return de_train, id_map

def preprocess_data(de_train, id_map):
    encoder = preprocessing.LabelEncoder()
    num_df = pd.DataFrame({
        'cell_type': encoder.fit_transform(de_train['cell_type']),
        'sm_name': encoder.fit_transform(de_train['sm_name']),
        'sm_lincs_id': encoder.fit_transform(de_train['sm_lincs_id']),
        'smiles': encoder.fit_transform(de_train['SMILES']),
        'control': de_train['control'].fillna(False).astype(int)
    })
    output_names = de_train.iloc[:, 5:].columns.tolist()
    df = pd.DataFrame({
        'cell_type': encoder.fit_transform(id_map['cell_type']),
        'sm_name': encoder.fit_transform(id_map['sm_name'])
    })
    return num_df, df, output_names

def train_knn_model(X_train, y_train):
    kn_model = KNeighborsRegressor()
    kn_model.fit(X_train, y_train)
    return kn_model

def perform_grid_search(kn_model, X_train, y_train):
    grid_params = {
        'n_neighbors': [4, 15, 30, 60, 100],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'metric': ['minkowski', 'euclidean', 'manhattan']
    }
    gs = GridSearchCV(kn_model, grid_params, verbose=1, cv=3, n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_

def make_predictions(model, X):
    return model.predict(X)

def main():
    de_train, id_map = load_data()
    num_df, df, output_names = preprocess_data(de_train, id_map)
    X_train, X_test, y_train, y_test = train_test_split(num_df.iloc[:, :2], de_train.iloc[:, 5:], test_size=0.3, shuffle=True)
    kn_model = train_knn_model(X_train, y_train)
    model = perform_grid_search(kn_model, X_train, y_train)
    predictions = make_predictions(model, df)
    test_out = make_predictions(model, X_test)
    mrrmse_score = np.sqrt(np.square(y_test - test_out).mean(axis=1))
    print(f' RMMSE = {mrrmse_score.mean()}')
    output = pd.DataFrame(predictions, index=id_map.index, columns=output_names)
    output.to_csv('./files/KNN_submission.csv')

if __name__ == '__main__':
    main()
