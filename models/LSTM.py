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
        data_x = np.array(data_x)
        train_x = data_x[:, 1:-1]
        train_y = data_x[:, -1]

        if self.is_model_created == False:
            self.create_model(train_x.shape[1])
            self.is_model_created = True

        train_x = self.sc_in.fit_transform(train_x)
        train_y = train_y.reshape(-1, 1)
        train_y = self.sc_out.fit_transform(train_y)
        train_x = np.array(train_x, dtype=float)
        train_y = np.array(train_y, dtype=float)
        train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
        self.model.fit(train_x, train_y, epochs=self.epochs, verbose=self.verbose, shuffle=False, batch_size=50)

    def predict(self, test_x):
        test_x = np.array(test_x.iloc[:, 1:], dtype=float)
        test_x = self.sc_in.transform(test_x)
        test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
        pred_y = self.model.predict(test_x)
        pred_y = pred_y.reshape(-1, 1)
        pred_y = self.sc_out.inverse_transform(pred_y)
        return pred_y

