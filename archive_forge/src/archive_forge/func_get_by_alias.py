import abc
import ctypes
import inspect
import json
import warnings
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from os import SEEK_END, environ
from os.path import getsize
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import scipy.sparse
from .compat import (PANDAS_INSTALLED, PYARROW_INSTALLED, arrow_cffi, arrow_is_floating, arrow_is_integer, concat,
from .libpath import find_lib_path
@classmethod
def get_by_alias(cls, *args) -> Set[str]:
    if cls.aliases is None:
        cls.aliases = cls._get_all_param_aliases()
    ret = set(args)
    for arg in args:
        for aliases in cls.aliases.values():
            if arg in aliases:
                ret.update(aliases)
                break
    return ret