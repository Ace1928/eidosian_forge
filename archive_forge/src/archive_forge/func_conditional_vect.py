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
def conditional_vect(function=None, **kwargs):
    """Use numba's vectorize decorator if numba is installed.

    Notes
    -----
        If called without arguments  then return wrapped function.
        @conditional_vect
        def my_func():
            return
        else called with arguments
        @conditional_vect(nopython=True)
        def my_func():
            return

    """

    def wrapper(function):
        try:
            numba = importlib.import_module('numba')
            return numba.vectorize(**kwargs)(function)
        except ImportError:
            return function
    if function:
        return wrapper(function)
    else:
        return wrapper