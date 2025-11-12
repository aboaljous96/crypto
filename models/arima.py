from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


# NOTE:
#   The original codebase exposed an ``arima`` factory at module level. Some
#   parts of the project – and, more importantly, a number of community forks –
#   still import ``arima`` or ``Arima`` from ``models.arima``.  The rewritten
#   module only provided the :class:`MyARIMA` class which broke those imports and
#   resulted in ``ImportError`` during the optimisation script bootstrap on
#   Windows.  We keep :class:`MyARIMA` as the main implementation but export
#   compatibility aliases so every import style continues to work.

class MyARIMA:
    sc_in = MinMaxScaler(feature_range=(0, 1))
    sc_out = MinMaxScaler(feature_range=(0, 1))

    def __init__(self, args):
        self.train_size = -1
        self.test_size = -1
        self.order = tuple(map(int, args.order.split(', ')))

    def fit(self, data_x):
        data_x = np.array(data_x)
        train_x = data_x[:, 1:-1]
        train_y = data_x[:, -1]
        print(train_x)
        self.train_size = train_x.shape[0]
        train_x = self.sc_in.fit_transform(train_x)
        train_y = train_y.reshape(-1, 1)
        train_y = self.sc_out.fit_transform(train_y)
        train_x = np.array(train_x, dtype=float)
        train_y = np.array(train_y, dtype=float)
        self.model = ARIMA(train_y,
                             exog=train_x,
                             order=self.order)
        self.result = self.model.fit()

    def predict(self, test_x):
        test_x = np.array(test_x.iloc[:, 1:], dtype=float)
        test_x = self.sc_in.transform(test_x)
        self.test_size = test_x.shape[0]
        pred_y = self.result.predict(start=self.train_size, end=self.train_size + self.test_size - 1, exog=test_x)
        # pred_y = self.result.predict(exog=test_x)
        pred_y = pred_y.reshape(-1, 1)
        pred_y = self.sc_out.inverse_transform(pred_y)
        return pred_y


# ---------------------------------------------------------------------------
# Backwards compatibility aliases
# ---------------------------------------------------------------------------
# Many legacy scripts expect ``arima``/``Arima`` callables.  Re-export the class
# under those names so ``from models.arima import arima`` keeps working and, at
# the same time, make the exported symbol list explicit.
Arima = MyARIMA
arima = MyARIMA

__all__ = ["MyARIMA", "Arima", "arima"]
