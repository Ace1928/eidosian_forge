import inspect
import warnings
import types
import collections
import itertools
from functools import lru_cache, wraps
from typing import Callable, List, Union, Iterable, TypeVar, cast
def no_escape_re_range_char(c):
    return c