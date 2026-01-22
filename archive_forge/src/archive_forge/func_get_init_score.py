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
def get_init_score(self) -> Optional[_LGBM_InitScoreType]:
    """Get the initial score of the Dataset.

        Returns
        -------
        init_score : list, list of lists (for multi-class task), numpy array, pandas Series, pandas DataFrame (for multi-class task), or None
            Init score of Booster.
            For a constructed ``Dataset``, this will only return ``None`` or a numpy array.
        """
    if self.init_score is None:
        self.init_score = self.get_field('init_score')
    return self.init_score