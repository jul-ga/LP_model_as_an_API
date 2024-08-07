from datetime import datetime

import dill
import pandas as pd
import copy

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC



def filter_data(df):
    df = df.copy()
    columns_to_drop = [
        'id',
        'url',
        'region',
        'region_url',
        'price',
        'manufacturer',
        'image_url',
        'description',
        'posting_date',
        'lat',
        'long'
    ]
    return df.drop(columns_to_drop, axis=1)


def outliers(df):
    df = df.copy()
    q25 = df['year'].quantile(0.25)
    q75 = df['year'].quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    df['year'] = df['year'].clip(lower=round(lower_bound), upper=round(upper_bound))
    return df


def create_new_predictors(df):
    import pandas as pd
    df = df.copy()
    df.loc[:, 'short_model'] = df['model'].apply(lambda x: x.lower().split(' ')[0] if not pd.isna(x) else x)
    df.loc[:, 'age_category'] = df['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))
    return df


def main():

    df = pd.read_csv('data/homework.csv')

    x = df.drop('price_category', axis=1)
    y = df['price_category']

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=object)
    unused_columns = [
        'year',
        'model',
        'fuel',
        'odometer',
        'title_status',
        'transmission',
        'state',
        'short_model',
        'age_category'
    ]

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features),
        ('drop_columns', 'drop', unused_columns)
    ])

    pipeline_template = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('outliers_year', FunctionTransformer(outliers)),
        ('new_pred', FunctionTransformer(create_new_predictors)),
        ('preprocessor', preprocessor),
    ])

    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        SVC()
    )

    results = []


    for model in models:
        pipe = copy.deepcopy(pipeline_template)
        pipe.steps.append(('classifier', model))

        score = cross_val_score(pipe, x, y, cv=4, scoring='accuracy', n_jobs=-1)
        mean_score = score.mean()
        results.append((pipe, mean_score, score.std()))

        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

    best_pipe, best_score, _ = max(results, key=lambda x: x[1])
    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')

    best_pipe.fit(x, y)
    with open('cars_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Car price prediction model',
                'author': 'user',
                'version': 1,
                'date': datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'accuracy': best_score
            }
        }, file, recurse=True)


if __name__ == '__main__':
    main()
