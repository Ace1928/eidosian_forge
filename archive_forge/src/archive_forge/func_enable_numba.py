import functools
import importlib
import importlib.resources
import re
import warnings
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis
from .rcparams import rcParams
@classmethod
def enable_numba(cls):
    """To enable numba."""
    if numba_check():
        cls.numba_flag = True
    else:
        raise ValueError('Numba is not installed')