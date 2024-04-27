import numpy as np 
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV        
        
        
de_train = pd.read_parquet('./data/de_train.parquet')
output_names = de_train.iloc[:,5:].columns.values.tolist()

id_map = pd.read_csv('/kaggle/input/open-problems-single-cell-perturbations/id_map.csv',index_col=0)
print(list(id_map['cell_type'].unique()))
print(len(list(id_map['sm_name'].unique())))

encoder = preprocessing.LabelEncoder()

num_df = pd.DataFrame()
num_df['cell_type'] = encoder.fit_transform(de_train['cell_type'])
num_df['sm_name'] = encoder.fit_transform(de_train['sm_name'])
num_df['sm_lincs_id'] =encoder.fit_transform(de_train['sm_lincs_id'])
num_df['smiles'] =encoder.fit_transform(de_train['SMILES'])
num_df['control'] = de_train['control'].fillna(False).astype('int')

df = pd.DataFrame()
df['cell_type'] = encoder.fit_transform(id_map['cell_type'])
df['sm_name'] = encoder.fit_transform(id_map['sm_name'])

X_train, X_test, y_train, y_test = train_test_split(num_df.iloc[:,:2], de_train.iloc[:,5:], test_size=0.3, shuffle=True)

kn_model = KNeighborsRegressor()
kn_model.fit(X_train, y_train)

grid_params = { 'n_neighbors' : [4,15, 30, 60, 100],
               'weights' : ['uniform','distance'],
               'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
               'metric' : ['minkowski','euclidean','manhattan']}

gs = GridSearchCV(kn_model, grid_params, verbose = 1, cv=3, n_jobs = -1)

g_res = gs.fit(X_train, y_train)

model = KNeighborsRegressor(metric='manhattan', 
                            n_neighbors=100, 
                            algorithm = 'brute',
                            weights='distance')
model.fit(X_train, y_train)

predictions = model.predict(df)
predictions.shape

test_out = model.predict(X_test)
mrrmse_score = np.sqrt(np.square(y_test - test_out).mean(axis=1))

output = pd.DataFrame(predictions, index=id_map.index, columns=output_names)
output.to_csv('submission.csv')