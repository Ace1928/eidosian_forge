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
def get_external_dmat(self) -> xgb.DMatrix:
    n_samples = self.X.shape[0]
    n_batches = 10
    per_batch = n_samples // n_batches + 1
    predictor = []
    response = []
    weight = []
    for i in range(n_batches):
        beg = i * per_batch
        end = min((i + 1) * per_batch, n_samples)
        assert end != beg
        X = self.X[beg:end, ...]
        y = self.y[beg:end]
        w = self.w[beg:end] if self.w is not None else None
        predictor.append(X)
        response.append(y)
        if w is not None:
            weight.append(w)
    it = IteratorForTest(predictor, response, weight if weight else None, cache='cache')
    return xgb.DMatrix(it)