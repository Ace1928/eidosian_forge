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
def replaces(arg):
    """Mark the method as replacing an entity in the collection.

        Adds "add to collection" and "remove from collection" handling to
        the method.  The decorator argument indicates which method argument
        holds the SQLAlchemy-relevant value to be added, and return value, if
        any will be considered the value to remove.

        Arguments can be specified positionally (i.e. integer) or by name::

            @collection.replaces(2)
            def __setitem__(self, index, item): ...

        """

    def decorator(fn):
        fn._sa_instrument_before = ('fire_append_event', arg)
        fn._sa_instrument_after = 'fire_remove_event'
        return fn
    return decorator