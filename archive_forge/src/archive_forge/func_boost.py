import copy
import ctypes
import importlib.util
import json
import os
import re
import sys
import warnings
import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import IntEnum, unique
from functools import wraps
from inspect import Parameter, signature
from typing import (
import numpy as np
import scipy.sparse
from ._typing import (
from .compat import PANDAS_INSTALLED, DataFrame, py_str
from .libpath import find_lib_path
def boost(self, dtrain: DMatrix, grad: np.ndarray, hess: np.ndarray) -> None:
    """Boost the booster for one iteration, with customized gradient
        statistics.  Like :py:func:`xgboost.Booster.update`, this
        function should not be called directly by users.

        Parameters
        ----------
        dtrain :
            The training DMatrix.
        grad :
            The first order of gradient.
        hess :
            The second order of gradient.

        """
    if len(grad) != len(hess):
        raise ValueError(f'grad / hess length mismatch: {len(grad)} / {len(hess)}')
    if not isinstance(dtrain, DMatrix):
        raise TypeError(f'invalid training matrix: {type(dtrain).__name__}')
    self._assign_dmatrix_features(dtrain)
    _check_call(_LIB.XGBoosterBoostOneIter(self.handle, dtrain.handle, c_array(ctypes.c_float, grad), c_array(ctypes.c_float, hess), c_bst_ulong(len(grad))))