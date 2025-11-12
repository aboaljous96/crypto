"""Training utilities for CryptoPredictions models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ensure_dataframe(dataset: pd.DataFrame | np.ndarray | list) -> pd.DataFrame:
    """Normalise a dataset into a clean ``pandas.DataFrame``.

    The project stores training samples as data frames containing a ``Date``
    column followed by flattened feature values and the ``prediction`` target
    as the final column.  Some callers occasionally pass a ``numpy`` array or a
    list of rows, therefore this helper converts the input into a dataframe and
    performs a minimal clean-up (type coercion, dropping missing values) so that
    models receive a consistent structure.
    """

    if dataset is None:
        raise ValueError("A valid dataset must be provided to the Trainer.")

    if isinstance(dataset, pd.DataFrame):
        frame = dataset.copy()
    else:
        frame = pd.DataFrame(dataset)

    if frame.empty:
        raise ValueError("Received an empty dataset for training.")

    # Normalise date column if present.  Keeping it as datetime simplifies
    # potential downstream feature engineering while models ignore it by
    # skipping the first column during fitting.
    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")

    # All other columns should be numeric; coercion ensures that stray
    # non-numeric strings turn into NaNs that can subsequently be dropped.
    for column in frame.columns:
        if column == "Date":
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Drop rows with any missing values to avoid issues in scikit-learn style
    # estimators that expect purely numeric matrices.
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    if frame.empty:
        raise ValueError("Dataset became empty after cleaning; cannot train model.")

    return frame


@dataclass
class Trainer:
    """Thin wrapper that prepares datasets before delegating to models."""

    args: object
    train_dataset: pd.DataFrame
    validation_dataset: Optional[pd.DataFrame]
    model: object

    def train(self) -> object:
        """Fit the underlying model on the provided dataset.

        The trainer is intentionally lightweight – most models in the project
        implement their own ``fit`` logic.  This method ensures the training
        frame is clean and numeric before invoking ``model.fit``.  The trained
        model instance is returned for convenience.
        """

        frame = _ensure_dataframe(self.train_dataset)
        logger.info("Training %s on %d samples.", self.model.__class__.__name__, len(frame))
        self.model.fit(frame)
        logger.info("Training finished for %s.", self.model.__class__.__name__)
        return self.model
