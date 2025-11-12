import logging
import math
import os
import os.path
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dateutil import parser
from tqdm import tqdm_notebook  # (Optional, used for progress-bars)

try:  # pragma: no-cover - exercised indirectly during runtime
    from bitmex import bitmex
except ImportError:  # pragma: no-cover - enables offline usage
    bitmex = None

# from binance.client import Client

from .creator import create_dataset, preprocess


logger = logging.getLogger(__name__)


class BitmexDataset:
    binsizes = {"1m": 1, "5m": 5, "1h": 60, "1d": 1440}

    def __init__(self, cfg):
        """Initialise the Bitmex dataset loader.

        The original implementation attempted to instantiate the BitMEX client
        unconditionally which fails in environments where the optional
        dependency is not installed (for example during offline evaluations or
        in CI).  We now keep the client lazy and rely on locally cached market
        data by default.  When the cache is disabled we fall back to the API.
        """

        self.cfg = cfg
        self.args = cfg.dataset_loader
        args = cfg.dataset_loader
        self.batch_size = args.batch_size
        self.symbol = args.symbol
        self.bin = args.binsize
        self.window_size = args.window_size
        self.features = args.features
        self.use_local_cache = getattr(args, "use_local_cache", True)
        self.local_cache_path = getattr(args, "local_cache_path", None)

        # Lazily created BitMEX client (only when the API is explicitly needed)
        self.bitmex_client = None
        self._bitmex_credentials = (
            getattr(args, "api_key", ""),
            getattr(args, "api_secret", ""),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_client(self):
        if self.bitmex_client is None:
            if bitmex is None:
                raise ImportError(
                    "The 'bitmex' package is required when use_local_cache is False."
                )
            api_key, api_secret = self._bitmex_credentials
            self.bitmex_client = bitmex(test=False, api_key=api_key, api_secret=api_secret)
        return self.bitmex_client

    def _resolve_cache_file(self, default_name: str) -> Optional[Path]:
        """Locate the requested cache file.

        The method looks in several sensible locations: an explicitly provided
        path, the project root (useful for repository fixtures) and the current
        working directory.  Returning :class:`pathlib.Path` keeps path handling
        platform agnostic.
        """

        candidates = []
        if self.local_cache_path:
            user_path = Path(self.local_cache_path)
            if user_path.is_dir():
                candidates.append(user_path / default_name)
            else:
                candidates.append(user_path)

        project_root = Path(__file__).resolve().parents[1]
        candidates.append(project_root / default_name)

        data_dir = Path(__file__).resolve().parent / "data"
        candidates.append(data_dir / default_name)

        candidates.append(Path.cwd() / default_name)

        searched_paths = []
        for candidate in candidates:
            searched_paths.append(str(candidate))
            if candidate.is_file():
                return candidate
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Cache lookup for %s exhausted candidates: %s",
                default_name,
                ", ".join(searched_paths),
            )
        return None

    ### FUNCTIONS
    def minutes_of_new_data(self, symbol, kline_size, data, source):
        self._ensure_client()

        if len(data) > 0:
            old = parser.parse(data["timestamp"].iloc[-1])
        elif source == "binance":
            old = datetime.strptime('1 Jan 2017', '%d %b %Y')
        elif source == "bitmex":
            client = self._ensure_client()
            old = client.Trade.Trade_getBucketed(
                symbol=symbol, binSize=kline_size, count=1, reverse=False
            ).result()[0][0]['timestamp']
        if source == "binance":
            new = pd.to_datetime(
                self.binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms'
            )
        if source == "bitmex":
            client = self._ensure_client()
            new = client.Trade.Trade_getBucketed(
                symbol=symbol, binSize=kline_size, count=1, reverse=True
            ).result()[0][0]['timestamp']
        return old, new

    def get_all_bitmex(self, symbol, kline_size, save=False):
        filename = f"{symbol}-{kline_size}-data.csv"

        if self.use_local_cache:
            cache_file = self._resolve_cache_file(filename)
            if cache_file is None:
                logger.warning(
                    "Local cache enabled but no file named '%s' was found. Falling back to the API.",
                    filename,
                )
            else:
                data_df = pd.read_csv(cache_file)
                data_df = data_df.rename({'timestamp': 'Date'}, axis=1)
                data_df['Date'] = pd.to_datetime(data_df['Date'])
                data_df = data_df.sort_values('Date').reset_index(drop=True)
                return preprocess(data_df, self.cfg)

        data_df = pd.DataFrame()
        client = self._ensure_client()
        oldest_point, newest_point = self.minutes_of_new_data(symbol, kline_size, data_df, source="bitmex")
        delta_min = (newest_point - oldest_point).total_seconds() / 60
        available_data = math.ceil(delta_min / self.binsizes[kline_size])
        rounds = math.ceil(available_data / self.batch_size)
        if rounds > 0:
            print(
                'Downloading %d minutes of new data available for %s, i.e. %d instances of %s data in %d rounds.'
                % (delta_min, symbol, available_data, kline_size, rounds)
            )
            for round_num in tqdm_notebook(range(rounds)):
                time.sleep(3)
                new_time = oldest_point + timedelta(
                    minutes=round_num * self.batch_size * self.binsizes[kline_size]
                )
                data = client.Trade.Trade_getBucketed(
                    symbol=symbol,
                    binSize=kline_size,
                    count=self.batch_size,
                    startTime=new_time,
                ).result()[0]
                temp_df = pd.DataFrame(data)
                data_df = pd.concat([data_df, temp_df], ignore_index=True)
        # data_df.set_index('Date', inplace=True)
        data_df = data_df.rename({'timestamp': 'Date'}, axis=1)
        data = preprocess(data_df, self.cfg)
        return data

    def create_dataset(self, df, window_size):
        dates = df['Date']
        df = df.drop('Date', axis=1)
        arr = np.array(df)
        data, profit_calculator = create_dataset(
            arr,
            list(dates),
            look_back=window_size,
            features=self.features,
            prediction_window=getattr(self.cfg.dataset_loader, 'prediction_window', 1)
        )
        return data, profit_calculator

    def get_dataset(self):
        dataset = self.get_all_bitmex(self.symbol, self.bin, save=True)
        return dataset




