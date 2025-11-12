import itertools
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import TimeSeriesSplit

from data_loader import get_dataset
from models import MODELS
from path_definition import HYDRA_PATH
from train import run_training


FEATURE_GROUPS: Dict[str, List[str]] = {
    "core": ["sma", "ema", "macd", "rsi", "bollinger"],
    "momentum_volatility": ["sma", "rsi", "atr", "obv", "chop"],
}

LSTM_GRID = {
    "window_size": [20, 40],
    "hidden_dim": [100, 150],
    "num_layers": [2, 3],
}

XGBOOST_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [4, 6],
}

OUTER_SPLITS = 3
INNER_SPLITS = 3
PREDICTION_WINDOW = 7

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BITMEX_CACHE = PROJECT_ROOT / "XBTUSD-1d-data.csv"


def clone_config(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


class AutoModelOptimizer:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.base_cfgs = self._create_base_configs()
        self.dataset_cache: Dict[Tuple[Tuple[str, ...], int, int], pd.DataFrame] = {}

    def _create_base_configs(self) -> Dict[str, DictConfig]:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        base_cfgs: Dict[str, DictConfig] = {}
        with initialize_config_dir(version_base=None, config_dir=HYDRA_PATH):
            for model_name in ("lstm", "xgboost"):
                cfg = compose(config_name="train", overrides=[f"model={model_name}", "dataset_loader=Bitmex"])
                cfg = clone_config(cfg)
                cfg.symbol = "XBTUSD"
                cfg.metrics = ["RMSE"]
                cfg.validation_method = "nested_cv"
                cfg.load_path = None
                cfg.save_dir = os.getcwd()
                cfg.dataset_loader.name = "Bitmex"
                cfg.dataset_loader.symbol = "XBTUSD"
                cfg.dataset_loader.binsize = "1d"
                cfg.dataset_loader.features = "Date, open, High, Low, close, volume"
                cfg.dataset_loader.train_start_date = "2016-01-01 00:00:00"
                cfg.dataset_loader.train_end_date = "2099-12-31 00:00:00"
                cfg.dataset_loader.valid_start_date = "2016-01-01 00:00:00"
                cfg.dataset_loader.valid_end_date = "2099-12-31 00:00:00"
                cfg.dataset_loader.prediction_window = 1
                cfg.dataset_loader.use_local_cache = True
                cfg.dataset_loader.local_cache_path = str(DEFAULT_BITMEX_CACHE)
                base_cfgs[model_name] = cfg
        return base_cfgs

    def _dataset_cache_key(self, indicators: List[str], window_size: int, prediction_window: int) -> Tuple[Tuple[str, ...], int, int]:
        return tuple(indicators), window_size, prediction_window

    def _prepare_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        dataset = df.copy()
        drop_columns = [col for col in ["predicted_high", "predicted_low"] if col in dataset.columns]
        if drop_columns:
            dataset = dataset.drop(columns=drop_columns)
        dataset = dataset.dropna().reset_index(drop=True)
        return dataset

    def _load_dataset(self, cfg: DictConfig, indicators: List[str], window_size: int, prediction_window: int) -> pd.DataFrame:
        cache_key = self._dataset_cache_key(indicators, window_size, prediction_window)
        if cache_key in self.dataset_cache:
            return self.dataset_cache[cache_key].copy()

        cfg_to_use = clone_config(cfg)
        cfg_to_use.dataset_loader.window_size = window_size
        cfg_to_use.dataset_loader.indicators_names = " ".join(indicators)
        cfg_to_use.dataset_loader.prediction_window = prediction_window

        dataset, _ = get_dataset(
            cfg_to_use.dataset_loader.name,
            cfg_to_use.dataset_loader.train_start_date,
            cfg_to_use.dataset_loader.valid_end_date,
            cfg_to_use,
        )
        dataset = self._prepare_dataset(dataset)
        dataset = dataset.sort_values("Date").reset_index(drop=True)
        self.dataset_cache[cache_key] = dataset
        return dataset.copy()

    def _resolve_splits(self, n_samples: int, desired: int) -> int:
        max_possible = max(0, n_samples - 2)
        if max_possible <= 1:
            return 0
        return min(desired, max_possible)

    def _train_and_predict(self, cfg: DictConfig, train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        model_args = clone_config(cfg.model)
        model = MODELS[model_args.type](model_args)
        model.fit(train_df)
        features = test_df.drop(columns=["prediction"])
        preds = model.predict(features)
        preds = np.array(preds).reshape(-1)
        return preds

    def _compute_rmse(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        predictions = np.array(predictions).reshape(-1)
        targets = np.array(targets).reshape(-1)
        return float(np.sqrt(np.mean(np.square(predictions - targets))))

    def _nested_cv_rmse(self, dataset: pd.DataFrame, cfg: DictConfig) -> Tuple[float, List[float]]:
        n_samples = len(dataset)
        outer_splits = self._resolve_splits(n_samples, OUTER_SPLITS)
        if outer_splits < 2:
            raise ValueError("Dataset is too small for nested cross validation.")

        outer_cv = TimeSeriesSplit(n_splits=outer_splits)
        outer_scores: List[float] = []

        for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(dataset)):
            train_df = dataset.iloc[train_idx].reset_index(drop=True)
            test_df = dataset.iloc[test_idx].reset_index(drop=True)

            inner_splits = self._resolve_splits(len(train_df), INNER_SPLITS)
            if inner_splits >= 2:
                inner_cv = TimeSeriesSplit(n_splits=inner_splits)
                inner_scores = []
                for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(train_df)):
                    inner_train = train_df.iloc[inner_train_idx].reset_index(drop=True)
                    inner_val = train_df.iloc[inner_val_idx].reset_index(drop=True)
                    preds = self._train_and_predict(cfg, inner_train, inner_val)
                    rmse = self._compute_rmse(preds, inner_val["prediction"].values)
                    inner_scores.append(rmse)
                    self.logger.debug(
                        "Inner fold %s | RMSE %.6f", f"{outer_fold}-{inner_fold}", rmse
                    )
                self.logger.debug(
                    "Outer fold %d | Inner mean RMSE %.6f", outer_fold, float(np.mean(inner_scores))
                )

            fold_preds = self._train_and_predict(cfg, train_df, test_df)
            fold_rmse = self._compute_rmse(fold_preds, test_df["prediction"].values)
            outer_scores.append(fold_rmse)
            self.logger.info(
                "Outer fold %d completed | RMSE %.6f", outer_fold, fold_rmse
            )

        return float(np.mean(outer_scores)), outer_scores

    def _build_config(self, base_cfg: DictConfig, model_type: str, params: Dict[str, int], indicators: List[str]) -> DictConfig:
        cfg = clone_config(base_cfg)
        cfg.model.type = model_type
        cfg.metrics = ["RMSE"]
        cfg.dataset_loader.indicators_names = " ".join(indicators)
        cfg.dataset_loader.prediction_window = 1

        if model_type == "lstm":
            cfg.model.hidden_dim = params["hidden_dim"]
            cfg.model.num_layers = params["num_layers"]
            cfg.dataset_loader.window_size = params["window_size"]
            cfg.model.verbose = 0
        elif model_type == "xgboost":
            cfg.model.n_estimators = params["n_estimators"]
            cfg.model.max_depth = params["max_depth"]
            cfg.dataset_loader.window_size = params.get("window_size", 40)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return cfg

    def run_search(self) -> Dict:
        results = []
        for group_name, indicators in FEATURE_GROUPS.items():
            for model_type, grid in (("lstm", LSTM_GRID), ("xgboost", XGBOOST_GRID)):
                grid_items = list(grid.items())
                keys, values = zip(*grid_items)
                for combo in itertools.product(*values):
                    params = dict(zip(keys, combo))
                    if model_type != "lstm":
                        params.setdefault("window_size", params.get("window_size", 40))
                    base_cfg = self.base_cfgs[model_type]
                    cfg = self._build_config(base_cfg, model_type, params, indicators)
                    dataset = self._load_dataset(
                        cfg,
                        indicators,
                        cfg.dataset_loader.window_size,
                        cfg.dataset_loader.prediction_window,
                    )
                    self.logger.info(
                        "Evaluating %s | Indicators: %s | Params: %s",
                        model_type.upper(),
                        group_name,
                        params,
                    )
                    mean_rmse, fold_scores = self._nested_cv_rmse(dataset, cfg)
                    result_entry = {
                        "model": model_type,
                        "indicator_group": group_name,
                        "indicators": list(indicators),
                        "params": params,
                        "rmse": mean_rmse,
                        "fold_scores": fold_scores,
                        "config": cfg,
                    }
                    results.append(result_entry)
                    self.logger.info(
                        "Finished %s | Indicators: %s | Mean RMSE: %.6f",
                        model_type.upper(),
                        group_name,
                        mean_rmse,
                    )
        if not results:
            raise RuntimeError("No experiments were executed during the search phase.")

        best_result = min(results, key=lambda item: item["rmse"])
        return best_result

    def _predict_with_horizon(self, cfg: DictConfig, indicators: List[str], window_size: int, horizon: int) -> float:
        dataset = self._load_dataset(cfg, indicators, window_size, horizon)
        model_cfg = clone_config(cfg)
        model_cfg.dataset_loader.window_size = window_size
        model_cfg.dataset_loader.prediction_window = horizon
        model_cfg.dataset_loader.indicators_names = " ".join(indicators)
        model = MODELS[model_cfg.model.type](clone_config(model_cfg.model))
        model.fit(dataset)
        features = dataset.drop(columns=["prediction"])
        preds = model.predict(features)
        preds = np.array(preds).reshape(-1)
        if preds.size == 0:
            raise ValueError("Model returned empty predictions for horizon {}".format(horizon))
        return float(preds[-1])

    def retrain_and_predict(self, best_result: Dict) -> List[float]:
        cfg = clone_config(best_result["config"])
        cfg.dataset_loader.prediction_window = PREDICTION_WINDOW

        # Ensure train.py pipeline is executed at least once
        train_cfg = clone_config(cfg)
        train_cfg.validation_method = "cross_validation"
        train_cfg.save_dir = tempfile.mkdtemp(prefix="auto_opt_train_")
        try:
            run_training(train_cfg)
        finally:
            shutil.rmtree(train_cfg.save_dir, ignore_errors=True)

        predictions = []
        indicators = best_result["indicators"]
        window_size = cfg.dataset_loader.window_size

        for day_ahead in range(1, PREDICTION_WINDOW + 1):
            prediction = self._predict_with_horizon(cfg, indicators, window_size, day_ahead)
            predictions.append(prediction)
        return predictions


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    optimizer = AutoModelOptimizer()
    best_result = optimizer.run_search()
    predictions = optimizer.retrain_and_predict(best_result)

    print("*** تحذير: التوقع المالي غير مؤكد ولا توجد ضمانات. استخدم هذه المعلومات على مسؤوليتك. ***")
    print(f"أفضل نموذج تم إيجاده: {best_result['model'].upper()}")
    print(f"أقل خطأ (RMSE): {best_result['rmse']:.6f}")
    print("\nالتوقعات لـ XBTUSD للأيام الـ 7 القادمة:")
    for idx, value in enumerate(predictions, start=1):
        print(f"اليوم {idx}: {value:.2f}")


if __name__ == "__main__":
    main()
