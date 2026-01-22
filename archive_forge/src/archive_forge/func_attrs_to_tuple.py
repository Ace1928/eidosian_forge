import contextlib
import copy
import enum
import functools
import inspect
import itertools
import linecache
import sys
import types
import typing
from operator import itemgetter
from . import _compat, _config, setters
from ._compat import (
from .exceptions import (
def attrs_to_tuple(obj):
    """
        Save us some typing.
        """
    return tuple((key(value) if key else value for value, key in ((getattr(obj, a.name), a.order_key) for a in attrs)))