from __future__ import annotations
import itertools
import logging
import random
import sys
from array import array
from packaging.version import parse as parse_version
from dask._compatibility import importlib_metadata
from dask.utils import Dispatch
@sizeof.register(cupy.ndarray)
def sizeof_cupy_ndarray(x):
    return int(x.nbytes)