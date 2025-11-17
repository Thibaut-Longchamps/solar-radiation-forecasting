import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sqlalchemy import create_engine
#from sklearn.preprocessing import MinMaxScaler

class WeatherPredictor:
    def __init__(self, model_path):
        # Load the trained model from file
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

        #self.scaler = MinMaxScaler(feature_range=(0, 1))

    def preprocess_data(self, data):
        # Ensure the date is in datetime format
        data['date'] = pd.to_datetime(data['date'])
        data['sunrise'] = pd.to_datetime(data['sunrise'])
        data['sunset'] = pd.to_datetime(data['sunset'])

        # Calculate daylight duration in hours
        data['daylight_duration'] = (data['sunset'] - data['sunrise']).dt.total_seconds() / 3600
        
        # Drop unnecessary columns
        columns_to_drop = ['sunrise', 'sunset']
        data = data.drop(columns=columns_to_drop, errors='ignore')

        # Extract year, month, day, hour, and minute from 'date' column
        data['date_year'] = data['date'].dt.year
        data['date_month'] = data['date'].dt.month
        data['date_day'] = data['date'].dt.day
        data['date_hour'] = data['date'].dt.hour
        data['date_minute'] = data['date'].dt.minute
        data['date_second'] = data['date'].dt.second

        # Drop the original 'date' column
        data = data.drop(columns=['date'])

        return data

    def predict(self, query):
        # Fetch and preprocess data
        #new_data = self.fetch_data(query)
        preprocessed_data = self.preprocess_data(query)

        # Make predictions using the loaded model
        predictions = self.model.predict(preprocessed_data)
        return predictions


    """def series_to_supervised(self, data, n_in=3, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        dff = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(dff.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)        n_vars = 1 if type(data) is list else data.shape[1]
        dff = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(dff.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(dff.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg"""
    
    # Preprocess the input, excluding GHI
    """def preprocess_input(self, data, n_in=1):
        '
        Prepares input data for inference by applying scaling and transformation into supervised format.
        data: raw input data as a list of lists or DataFrame with shape (1, n_features - GHI)
        '
        # Ensure the date is in datetime format
        data['date'] = pd.to_datetime(data['date'])
        data['sunrise'] = pd.to_datetime(data['sunrise'])
        data['sunset'] = pd.to_datetime(data['sunset'])

        # Calculate daylight duration in hours
        data['daylight_duration'] = (data['sunset'] - data['sunrise']).dt.total_seconds() / 3600
        data.set_index('date', inplace=True)
        # Drop unnecessary columns
        columns_to_drop = ['sunrise', 'sunset']
        data = data.drop(columns=columns_to_drop, errors='ignore')
        ## resampling of data over hour
        df_resample = data.resample('h').mean()
        # Convert the index to a regular column if it's a DateTimeIndex
        df_resample = df_resample.reset_index()
        # Select only the numeric columns
        numeric_columns = df_resample.select_dtypes(include=['number']).columns
        df_resample = df_resample[numeric_columns]

        values = df_resample.values
        

        scaled_data = self.scaler.fit_transform(values)
        

        # Convert to supervised format, without GHI
        reframed_data = self.series_to_supervised(scaled_data, n_in=n_in, n_out=1)
        # Drop extra columns to match the training model requirements
        # This may need adjusting based on your final model
        
        reframed_data.drop(reframed_data.columns[[11, 12, 13, 14, 15, 16, 17, 18, 19]], axis=1, inplace=True)

        return reframed_data.values"""
    
    """def predixt_lstm(self, df):

        prepared_input = self.preprocess_input(df)
        prepared_input = prepared_input.reshape((prepared_input.shape[0], 1, prepared_input.shape[1]))
        # Make prediction
        yhat = self.model.predict(prepared_input)
        prepared_input = prepared_input.reshape((prepared_input.shape[0], 11))
        inv_yhat = np.concatenate((yhat, prepared_input[:, -10:]), axis=1)
        inv_yhat = self.scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]

        return inv_yhat"""


