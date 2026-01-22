import inspect
import warnings
import types
import collections
import itertools
from functools import lru_cache, wraps
from typing import Callable, List, Union, Iterable, TypeVar, cast
def replaced_by_pep8(fn: C) -> Callable[[Callable], C]:
    """
    Decorator for pre-PEP8 compatibility synonyms, to link them to the new function.
    """
    return lambda other: _make_synonym_function(other.__name__, fn)