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
def predictor_equal(lhs: xgb.DMatrix, rhs: xgb.DMatrix) -> bool:
    """Assert whether two DMatrices contain the same predictors."""
    lcsr = lhs.get_data()
    rcsr = rhs.get_data()
    return all((np.array_equal(lcsr.data, rcsr.data), np.array_equal(lcsr.indices, rcsr.indices), np.array_equal(lcsr.indptr, rcsr.indptr)))