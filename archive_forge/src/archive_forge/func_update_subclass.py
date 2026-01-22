from __future__ import annotations
import collections
from itertools import chain
import threading
from types import TracebackType
import typing
from typing import Any
from typing import cast
from typing import Collection
from typing import Deque
from typing import FrozenSet
from typing import Generic
from typing import Iterator
from typing import MutableMapping
from typing import MutableSequence
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import weakref
from . import legacy
from . import registry
from .registry import _ET
from .registry import _EventKey
from .registry import _ListenerFnType
from .. import exc
from .. import util
from ..util.concurrency import AsyncAdaptedLock
from ..util.typing import Protocol
def update_subclass(self, target: Type[_ET]) -> None:
    if target not in self._clslevel:
        if getattr(target, '_sa_propagate_class_events', True):
            self._clslevel[target] = collections.deque()
        else:
            self._clslevel[target] = _empty_collection()
    clslevel = self._clslevel[target]
    cls: Type[_ET]
    for cls in target.__mro__[1:]:
        if cls in self._clslevel:
            clslevel.extend([fn for fn in self._clslevel[cls] if fn not in clslevel])