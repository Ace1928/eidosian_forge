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
def create_proxy_methods(target_cls: Type[Any], target_cls_sphinx_name: str, proxy_cls_sphinx_name: str, classmethods: Sequence[str]=(), methods: Sequence[str]=(), attributes: Sequence[str]=(), use_intermediate_variable: Sequence[str]=()) -> Callable[[_T], _T]:
    """A class decorator indicating attributes should refer to a proxy
    class.

    This decorator is now a "marker" that does nothing at runtime.  Instead,
    it is consumed by the tools/generate_proxy_methods.py script to
    statically generate proxy methods and attributes that are fully
    recognized by typing tools such as mypy.

    """

    def decorate(cls):
        return cls
    return decorate