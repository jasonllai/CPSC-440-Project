import pandas as pd

de_train = pd.read_parquet('./data/de_train.parquet')
print(de_train.head())