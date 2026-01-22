import io
import re
import functools
import inspect
import os
import sys
import numbers
import warnings
from pathlib import Path, PurePath
from typing import (
from ase.atoms import Atoms
from importlib import import_module
from ase.parallel import parallel_function, parallel_generator
def index2range(index, length):
    """Convert slice or integer to range.

    If index is an integer, range will contain only that integer."""
    obj = range(length)[index]
    if isinstance(obj, numbers.Integral):
        obj = range(obj, obj + 1)
    return obj