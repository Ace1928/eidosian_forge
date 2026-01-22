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
def string2index(string: str) -> Union[int, slice, str]:
    """Convert index string to either int or slice"""
    if ':' not in string:
        try:
            return int(string)
        except ValueError:
            return string
    i: List[Optional[int]] = []
    for s in string.split(':'):
        if s == '':
            i.append(None)
        else:
            i.append(int(s))
    i += (3 - len(i)) * [None]
    return slice(*i)