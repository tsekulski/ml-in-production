from sklearn.base import BaseEstimator, TransformerMixin # for definition of custom transformers
import numpy as np
import pandas as pd
import calendar
#import bisect
#import warnings
#import json


# Custom transformer for filling missing values in the categorical "saving accounts" and "checking account" columns
class MissingValFiller(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols_to_fill = None

    def fit(self, features_df, labels_df=None, columns_to_fill=None):
        if columns_to_fill is None:
            self.cols_to_fill = ['Saving accounts', 'Checking account']
        else:
            self.cols_to_fill = columns_to_fill

        return self

    def transform(self, features_df):
        transformed_df = features_df.copy()
        for column in transformed_df[self.cols_to_fill]:
            transformed_df[column] = transformed_df[column].fillna('Missing')

        return transformed_df

# Custom transformer for casting "Job" column to string
class StringCaster(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols_to_cast = None

    def fit(self, features_df, labels_df=None, columns_to_cast=None):
        if columns_to_cast is None:
            self.cols_to_cast = ['Job']
        else:
            self.cols_to_cast = columns_to_cast

        return self

    def transform(self, features_df):
        transformed_df = features_df.copy()
        for column in transformed_df[self.cols_to_cast]:
            transformed_df[column] = transformed_df[column].astype(str)

        return transformed_df

# Custom transformer for engineering "Age" feature
class AgeEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.reference_date_column = None
        self.birth_date_column = None

    def fit(self, features_df, labels_df=None, r_date_col=None, b_date_col=None):
        if r_date_col is None or b_date_col is None:
            self.reference_date_column = 'Purchase_date'
            self.birth_date_column = 'Birth_date'
        else:
            self.reference_date_column = r_date_col
            self.birth_date_column = b_date_col

        return self

    def transform(self, features_df):
        transformed_df = features_df.copy()
        transformed_df['Age'] = transformed_df.apply(lambda row: self.create_age_column(row), axis=1)
        transformed_df.drop([self.birth_date_column], axis=1, inplace=True)

        return transformed_df

    def calculate_age(self, referenceDate, birthDate):
        days_in_year = 365.2425
        try:
            age = int((pd.to_datetime(referenceDate, infer_datetime_format=True) - pd.to_datetime(str(birthDate), infer_datetime_format=True)).days / days_in_year)
        except:
            age = None

        return age

    def create_age_column(self, row):
        return self.calculate_age(row[self.reference_date_column], row[self.birth_date_column])


# Customer transformer for engineering "Day of week" feature
class DayOfWeekEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.date_column = None

    def fit(self, features_df, labels_df=None, date_col=None):
        if date_col is None:
            self.date_column = 'Purchase_date'
        else:
            self.date_column = date_col

        return self

    def transform(self, features_df):
        transformed_df = features_df.copy()
        transformed_df['Weekday'] = transformed_df[self.date_column].map(lambda x: self.find_day_of_week(x))
        transformed_df.drop([self.date_column], axis=1, inplace=True)

        return transformed_df

    def find_day_of_week(self, string):
        recognized_date = pd.to_datetime(string, infer_datetime_format=True)

        return calendar.day_name[recognized_date.weekday()]


# Custom transformer for engineering binary iPhone-Other device feature
class DeviceEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.device_column = None

    def fit(self, features_df, labels_df=None, device_col=None):
        if device_col is None:
            self.device_column = 'Device'
        else:
            self.device_column = device_col

        return self

    def transform(self, features_df):
        transformed_df = features_df.copy()
        transformed_df.loc[:, self.device_column] = transformed_df[self.device_column].map(
            lambda x: self.identify_device(str(x)))

        return transformed_df

    def identify_device(self, string):
        if 'iphone' in string.lower():
            return 'iPhone'
        else:
            return 'Other'


# Transformer for preceding and trailing whitespaces
class WhitespaceRemover(BaseEstimator, TransformerMixin):
    # def __init__(self):
    # no variables to be initialized

    def remove_whitespace(self, features_df):
        for col in features_df:
            if features_df[col].dtype == 'object':
                features_df[col] = features_df[col].str.strip()
        return features_df

    def fit(self, features_df=None, labels_df=None):
        return self

    def transform(self, features_df):
        transformed_df = features_df.copy()
        transformed_df = self.remove_whitespace(transformed_df)

        return transformed_df


# Transformer for casting data into types present in original training data
class OriginalDtypesCaster(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.column_types = {}

    def fit(self, features_df, labels_df=None):
        dtypes = features_df.dtypes
        dtypes_col = dtypes.index
        dtypes_type = [i.name for i in dtypes.values]
        self.column_types = dict(zip(dtypes_col, dtypes_type))

        return self

    def transform(self, features_df):
        transformed_df = features_df.copy()
        transformed_df = transformed_df.astype(self.column_types)

        return transformed_df


# Transformer for transforming df to a dict (needed input format for DictVectorizer)
class DfDict(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, features_df, labels_df=None):
        return self

    def transform(self, features_df):
        return features_df.to_dict('records')
