from __future__ import annotations
import itertools
import logging
import random
import sys
from array import array
from packaging.version import parse as parse_version
from dask._compatibility import importlib_metadata
from dask.utils import Dispatch
@sizeof.register(dict)
def sizeof_python_dict(d):
    return sys.getsizeof(d) + sizeof(list(d.keys())) + sizeof(list(d.values())) - 2 * sizeof(list())