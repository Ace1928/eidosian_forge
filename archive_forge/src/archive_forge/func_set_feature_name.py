import abc
import ctypes
import inspect
import json
import warnings
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from os import SEEK_END, environ
from os.path import getsize
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import scipy.sparse
from .compat import (PANDAS_INSTALLED, PYARROW_INSTALLED, arrow_cffi, arrow_is_floating, arrow_is_integer, concat,
from .libpath import find_lib_path
def set_feature_name(self, feature_name: _LGBM_FeatureNameConfiguration) -> 'Dataset':
    """Set feature name.

        Parameters
        ----------
        feature_name : list of str
            Feature names.

        Returns
        -------
        self : Dataset
            Dataset with set feature name.
        """
    if feature_name != 'auto':
        self.feature_name = feature_name
    if self._handle is not None and feature_name is not None and (feature_name != 'auto'):
        if len(feature_name) != self.num_feature():
            raise ValueError(f"Length of feature_name({len(feature_name)}) and num_feature({self.num_feature()}) don't match")
        c_feature_name = [_c_str(name) for name in feature_name]
        _safe_call(_LIB.LGBM_DatasetSetFeatureNames(self._handle, _c_array(ctypes.c_char_p, c_feature_name), ctypes.c_int(len(feature_name))))
    return self