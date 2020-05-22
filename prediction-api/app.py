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
    meta_data_columns = ['Job',
                         'Housing',
                         'Saving accounts',
                         'Checking account',
                         'Credit amount',
                         'Duration',
                         'Purpose',
                         'Gender',
                         'Purchase_date',
                         'Birth_date',
                         'Device']

    def order_data(data_):
        df = pd.DataFrame.from_dict(data_,
                                    dtype='object')  # this prevents pandas from reading e.g. '0' as float '0.0'
        df = df[meta_data_columns].copy()
        df = df.reindex(columns=meta_data_columns)
        return df

    json_ = request.json  # this loads json data into a dict called json_
    print(json_)
    query_df = order_data(json_)

    print(query_df)
    print(query_df.isna().sum())
    prediction = e2e_pipeline.predict_proba(query_df)[:,1]

    return jsonify(prediction.tolist())


if __name__ == '__main__':
    e2e_pipeline = joblib.load('e2e_pipeline.pkl')
    app.run(debug=True, host='0.0.0.0')
