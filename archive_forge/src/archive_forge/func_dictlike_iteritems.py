from __future__ import annotations
import collections
import enum
from functools import update_wrapper
import inspect
import itertools
import operator
import re
import sys
import textwrap
import threading
import types
from types import CodeType
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import warnings
from . import _collections
from . import compat
from ._has_cy import HAS_CYEXTENSION
from .typing import Literal
from .. import exc
def dictlike_iteritems(dictlike):
    """Return a (key, value) iterator for almost any dict-like object."""
    if hasattr(dictlike, 'items'):
        return list(dictlike.items())
    getter = getattr(dictlike, '__getitem__', getattr(dictlike, 'get', None))
    if getter is None:
        raise TypeError("Object '%r' is not dict-like" % dictlike)
    if hasattr(dictlike, 'iterkeys'):

        def iterator():
            for key in dictlike.iterkeys():
                assert getter is not None
                yield (key, getter(key))
        return iterator()
    elif hasattr(dictlike, 'keys'):
        return iter(((key, getter(key)) for key in dictlike.keys()))
    else:
        raise TypeError("Object '%r' is not dict-like" % dictlike)