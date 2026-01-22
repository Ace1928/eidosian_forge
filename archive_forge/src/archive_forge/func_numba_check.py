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
def numba_check():
    """Check if numba is installed."""
    numba = importlib.util.find_spec('numba')
    return numba is not None