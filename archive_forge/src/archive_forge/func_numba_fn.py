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
@lazy_property
def numba_fn(self):
    """Memoized compiled function."""
    try:
        numba = importlib.import_module('numba')
        numba_fn = numba.jit(**self.kwargs)(self.function)
    except ImportError:
        numba_fn = self.function
    return numba_fn