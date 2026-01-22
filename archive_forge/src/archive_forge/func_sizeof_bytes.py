from __future__ import annotations
import itertools
import logging
import random
import sys
from array import array
from packaging.version import parse as parse_version
from dask._compatibility import importlib_metadata
from dask.utils import Dispatch
@sizeof.register(bytes)
@sizeof.register(bytearray)
def sizeof_bytes(o):
    return len(o)