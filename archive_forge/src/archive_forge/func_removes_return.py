from __future__ import annotations
import operator
import threading
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from .base import NO_KEY
from .. import exc as sa_exc
from .. import util
from ..sql.base import NO_ARG
from ..util.compat import inspect_getfullargspec
from ..util.typing import Protocol
@staticmethod
def removes_return():
    """Mark the method as removing an entity in the collection.

        Adds "remove from collection" handling to the method.  The return
        value of the method, if any, is considered the value to remove.  The
        method arguments are not inspected::

            @collection.removes_return()
            def pop(self): ...

        For methods where the value to remove is known at call-time, use
        collection.remove.

        """

    def decorator(fn):
        fn._sa_instrument_after = 'fire_remove_event'
        return fn
    return decorator