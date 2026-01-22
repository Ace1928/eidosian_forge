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
def removes(arg):
    """Mark the method as removing an entity in the collection.

        Adds "remove from collection" handling to the method.  The decorator
        argument indicates which method argument holds the SQLAlchemy-relevant
        value to be removed. Arguments can be specified positionally (i.e.
        integer) or by name::

            @collection.removes(1)
            def zap(self, item): ...

        For methods where the value to remove is not known at call-time, use
        collection.removes_return.

        """

    def decorator(fn):
        fn._sa_instrument_before = ('fire_remove_event', arg)
        return fn
    return decorator