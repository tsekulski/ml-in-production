from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import json
import calendar

from sklearn.base import BaseEstimator, TransformerMixin
from src.pipeline_components import MissingValFiller, StringCaster, AgeEncoder, DayOfWeekEncoder, DeviceEncoder
from src.pipeline_components import WhitespaceRemover, OriginalDtypesCaster, DfDict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    def order_data(data_):
        df = pd.DataFrame.from_dict(data_,
                                    dtype='object') # this prevents pandas from reading e.g. '0' as float '0.0'

        df = df[column_names].copy()
        df = df.reindex(columns=column_names)
        return df

    def cast_to_original_dtypes(df):
        for col in column_names:
            dtype = column_dtypes[col]
            df[col] = df[col].astype(dtype)

        return df

    json_ = request.json  # this loads json data into a dict called json_
    query_df = order_data(json_)
    cast_df = cast_to_original_dtypes(query_df)
    prediction = e2e_pipeline.predict_proba(cast_df)[:,1]

    return jsonify(prediction.tolist())


if __name__ == '__main__':
    e2e_pipeline = joblib.load('e2e_pipeline.pkl')
    column_names = []
    with open('column_names.txt', 'r') as f:
        for line in f:
            column_names.append(line.strip())

    with open('column_dtypes.json', 'r') as fp:
        column_dtypes = json.load(fp)

    app.run(debug=True, host='0.0.0.0')
