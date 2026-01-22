from __future__ import annotations
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import base
from . import collections
from . import exc
from . import interfaces
from . import state
from ._typing import _O
from .attributes import _is_collection_attribute_impl
from .. import util
from ..event import EventTarget
from ..util import HasMemoized
from ..util.typing import Literal
from ..util.typing import Protocol
def uninstrument_attribute(self, key, propagated=False):
    if key not in self:
        return
    if propagated:
        if key in self.local_attrs:
            return
    else:
        del self.local_attrs[key]
        self.uninstall_descriptor(key)
    self._reset_memoizations()
    del self[key]
    for cls in self.class_.__subclasses__():
        manager = opt_manager_of_class(cls)
        if manager:
            manager.uninstrument_attribute(key, True)