import gc
import importlib.util
import multiprocessing
import os
import platform
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import StringIO
from platform import system
from typing import (
import numpy as np
import pytest
from scipy import sparse
import xgboost as xgb
from xgboost.core import ArrayLike
from xgboost.sklearn import SklObjective
from xgboost.testing.data import (
from hypothesis import strategies
from hypothesis.extra.numpy import arrays
def make_ltr(n_samples: int, n_features: int, n_query_groups: int, max_rel: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make a dataset for testing LTR."""
    rng = np.random.default_rng(1994)
    X = rng.normal(0, 1.0, size=n_samples * n_features).reshape(n_samples, n_features)
    y = np.sum(X, axis=1)
    y -= y.min()
    y = np.round(y / y.max() * max_rel).astype(np.int32)
    qid = rng.integers(0, n_query_groups, size=n_samples, dtype=np.int32)
    w = rng.normal(0, 1.0, size=n_query_groups)
    w -= np.min(w)
    w /= np.max(w)
    qid = np.sort(qid)
    return (X, y, qid, w)