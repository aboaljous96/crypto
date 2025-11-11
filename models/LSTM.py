import numpy as np
import pandas as pd

import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation, Dense,Dropout

from sklearn.preprocessing import MinMaxScaler


class MyLSTM:
    sc_in = MinMaxScaler(feature_range=(0, 1))
    sc_out = MinMaxScaler(feature_range=(0, 1))

    def __init__(self, args):
        self.model = Sequential()
        self.is_model_created = False
        self.hidden_dim = getattr(args, 'hidden_dim', 256)
        self.epochs = getattr(args, 'epochs', 50)
        self.num_layers = getattr(args, 'num_layers', 2)
        self.dropout = getattr(args, 'dropout', 0.0)
        self.verbose = getattr(args, 'verbose', 1)


    def create_model(self, shape_):
        self.model = Sequential()
        for layer_idx in range(self.num_layers):
            return_sequences = layer_idx < self.num_layers - 1
            if layer_idx == 0:
                self.model.add(LSTM(self.hidden_dim, return_sequences=return_sequences, input_shape=(1, shape_)))
            else:
                self.model.add(LSTM(self.hidden_dim, return_sequences=return_sequences))
            if self.dropout > 0 and (return_sequences or layer_idx == self.num_layers - 1):
                self.model.add(Dropout(self.dropout))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def fit(self, data_x):
        # data_x هي DataFrame. لن نحولها إلى numpy array مباشرة.
        
        # --- [التصحيح] ---
        # 1. استخدم .iloc لتقسيم الـ DataFrame أولاً (لتجنب عمود 'Date')
        train_x_df = data_x.iloc[:, 1:-1]
        train_y_series = data_x.iloc[:, -1]

        # 2. الآن قم بالتحويل إلى numpy array بأمان
        train_x = np.array(train_x_df, dtype=float)
        train_y = np.array(train_y_series, dtype=float)
        # --- [نهاية التصحيح] ---


        if self.is_model_created == False:
            self.create_model(train_x.shape[1])
            self.is_model_created = True

        train_x = self.sc_in.fit_transform(train_x) # <--- ستعمل الآن
        train_y = train_y.reshape(-1, 1)
        train_y = self.sc_out.fit_transform(train_y)
        
        # هذه الأسطر لم نعد بحاجة لها هنا
        # train_x = np.array(train_x, dtype=float)
        # train_y = np.array(train_y, dtype=float)
        
        train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
        self.model.fit(train_x, train_y, epochs=self.epochs, verbose=self.verbose, shuffle=False, batch_size=50)

    def predict(self, test_x):
        # دالة التوقع كانت صحيحة وتستخدم .iloc
        test_x_features = test_x.iloc[:, 1:]
        test_x_array = np.array(test_x_features, dtype=float)
        
        test_x_scaled = self.sc_in.transform(test_x_array)
        test_x_reshaped = np.reshape(test_x_scaled, (test_x_scaled.shape[0], 1, test_x_scaled.shape[1]))
        
        pred_y = self.model.predict(test_x_reshaped)
        pred_y = pred_y.reshape(-1, 1)
        pred_y = self.sc_out.inverse_transform(pred_y)
        return pred_y
