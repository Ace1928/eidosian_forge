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
def only_once(fn: Callable[..., _T], retry_on_exception: bool) -> Callable[..., Optional[_T]]:
    """Decorate the given function to be a no-op after it is called exactly
    once."""
    once = [fn]

    def go(*arg: Any, **kw: Any) -> Optional[_T]:
        strong_fn = fn
        if once:
            once_fn = once.pop()
            try:
                return once_fn(*arg, **kw)
            except:
                if retry_on_exception:
                    once.insert(0, once_fn)
                raise
        return None
    return go