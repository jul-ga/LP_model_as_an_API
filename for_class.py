import pandas as pd

df = pd.read_csv('model/data/homework.csv').drop('price_category', axis=1)


types = {
    'int64': 'int',
    'float64': 'float'
}
for k, v in df.dtypes.items():
    print(f'{k}: {types.get(str(v), "str")}')