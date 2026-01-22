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
def duck_type_collection(specimen: Any, default: Optional[Type[Any]]=None) -> Optional[Type[Any]]:
    """Given an instance or class, guess if it is or is acting as one of
    the basic collection types: list, set and dict.  If the __emulates__
    property is present, return that preferentially.
    """
    if hasattr(specimen, '__emulates__'):
        if specimen.__emulates__ is not None and issubclass(specimen.__emulates__, set):
            return set
        else:
            return specimen.__emulates__
    isa = issubclass if isinstance(specimen, type) else isinstance
    if isa(specimen, list):
        return list
    elif isa(specimen, set):
        return set
    elif isa(specimen, dict):
        return dict
    if hasattr(specimen, 'append'):
        return list
    elif hasattr(specimen, 'add'):
        return set
    elif hasattr(specimen, 'set'):
        return dict
    else:
        return default