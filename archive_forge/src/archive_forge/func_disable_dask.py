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
def disable_dask(cls):
    """To disable Dask."""
    cls.dask_flag = False
    cls.dask_kwargs = None