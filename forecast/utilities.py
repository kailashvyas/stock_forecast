import quandl, math
import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class Utilities():

    def pre_process(self, df):
        df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
        df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
        df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

        df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
        forecast_col = 'Adj. Close'
        df.fillna(value=-99999, inplace=True)
        forecast_out = int(math.ceil(0.01 * len(df)))
        df['label'] = df[forecast_col].shift(-forecast_out)
        self.df_processed = df

        X = np.array(df.drop(['label'], 1))
        X = preprocessing.scale(X)
        self.X_lately = X[-forecast_out:]
        X = X[:-forecast_out]

        df.dropna(inplace=True)

        y = np.array(df['label'])

        return train_test_split(X, y, test_size=0.2)

    def get_data(self, company):
        df = quandl.get("WIKI/"+company)

        return df

    def get_recent(self):

        return self.X_lately

    def get_formatted_data(self):

        data = []
        for i, row in enumerate(self.df_processed[['Adj. Close']].values):
            date = self.df_processed.index[i]
            date = datetime.datetime.fromtimestamp(date.timestamp())
            data.append([date, int(row[0])])

        return data

    def get_predicted_forecast(self, forecast_set):

        last_date = self.df_processed.iloc[-1].name
        last_unix = last_date.timestamp()
        one_day = 86400
        next_unix = last_unix + one_day
        predicted_forecast = []
        for i in forecast_set:
            next_date = datetime.datetime.fromtimestamp(next_unix)
            predicted_forecast.append([next_date, int(i)])
            next_unix += 86400

        return predicted_forecast




