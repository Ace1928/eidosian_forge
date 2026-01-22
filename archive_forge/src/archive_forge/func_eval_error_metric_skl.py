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
def eval_error_metric_skl(y_true: np.ndarray, y_score: np.ndarray) -> np.float64:
    """Evaluation metric that looks like metrics provided by sklearn."""
    r = np.zeros(y_score.shape)
    gt = y_score > 0.5
    r[gt] = 1 - y_true[gt]
    le = y_score <= 0.5
    r[le] = y_true[le]
    return np.sum(r)