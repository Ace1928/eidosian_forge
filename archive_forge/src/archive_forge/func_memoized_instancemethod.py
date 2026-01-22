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
@classmethod
def memoized_instancemethod(cls, fn: _F) -> _F:
    """Decorate a method memoize its return value.

        :meta private:

        """

    def oneshot(self: Any, *args: Any, **kw: Any) -> Any:
        result = fn(self, *args, **kw)

        def memo(*a, **kw):
            return result
        memo.__name__ = fn.__name__
        memo.__doc__ = fn.__doc__
        self.__dict__[fn.__name__] = memo
        self._memoized_keys |= {fn.__name__}
        return result
    return update_wrapper(oneshot, fn)