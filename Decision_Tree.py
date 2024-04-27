import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split




de_train = pd.read_parquet('./data/de_train.parquet')
output_names = de_train.iloc[:,5:].columns.values.tolist()
# print(list(de_train['cell_type'].unique()))
# print(len(list(de_train['sm_name'].unique())))

id_map = pd.read_csv('./data/id_map.csv',index_col=0)
# print(list(id_map['cell_type'].unique()))
# print(len(list(id_map['sm_name'].unique())))

encoder = preprocessing.LabelEncoder()

df = pd.DataFrame()
df['cell_type'] = encoder.fit_transform(id_map['cell_type'])
df['sm_name'] = encoder.fit_transform(id_map['sm_name'])

num_df = pd.DataFrame()
num_df['cell_type'] = encoder.fit_transform(de_train['cell_type'])
num_df['sm_name'] = encoder.fit_transform(de_train['sm_name'])
num_df['sm_lincs_id'] =encoder.fit_transform(de_train['sm_lincs_id'])
num_df['smiles'] =encoder.fit_transform(de_train['SMILES'])
num_df['control'] = de_train['control'].fillna(False).astype('int')



X_train, X_test, y_train, y_test = train_test_split(num_df.iloc[:,:2], de_train.iloc[:,5:], test_size=0.3, shuffle=True)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# print(f"Train accuracy = {model.score(X_train,y_train) * 100}%")
# print(f"Test accuracy = {model.score(X_test,y_test) * 100}%")

predictions = model.predict(df)

test_out = model.predict(X_test)
mrrmse_score = np.sqrt(np.square(y_test - test_out).mean(axis=1))
print(f' RMMSE = {mrrmse_score.mean()}')



plt.scatter(list(range(0, 185)), mrrmse_score)
plt.xlabel('Iterations')
plt.ylabel('MRRMSE')
plt.show()

output = pd.DataFrame(predictions, index=id_map.index, columns=output_names)
print(output.head())

# train on whole dataset
X_train, y_train = num_df.iloc[:,:2], de_train.iloc[:,5:]
model.fit(X_train, y_train)
predictions = model.predict(df)
output = pd.DataFrame(predictions, index=id_map.index, columns=output_names)

output.to_csv('./files/DT_submission.csv')

# submit
# from submit_to_kaggle import submit_to_kaggle
# submit_to_kaggle(output)