from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import instrumentation
from . import interfaces
from . import mapperlib
from .attributes import QueryableAttribute
from .base import _mapper_or_none
from .base import NO_KEY
from .instrumentation import ClassManager
from .instrumentation import InstrumentationFactory
from .query import BulkDelete
from .query import BulkUpdate
from .query import Query
from .scoping import scoped_session
from .session import Session
from .session import sessionmaker
from .. import event
from .. import exc
from .. import util
from ..event import EventTarget
from ..event.registry import _ET
from ..util.compat import inspect_getfullargspec
def init_failure(self, target: _O, args: Any, kwargs: Any) -> None:
    """Receive an instance when its constructor has been called,
        and raised an exception.

        This method is only called during a userland construction of
        an object, in conjunction with the object's constructor, e.g.
        its ``__init__`` method. It is not called when an object is loaded
        from the database.

        The event is invoked after an exception raised by the ``__init__``
        method is caught.  After the event
        is invoked, the original exception is re-raised outwards, so that
        the construction of the object still raises an exception.   The
        actual exception and stack trace raised should be present in
        ``sys.exc_info()``.

        :param target: the mapped instance.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :param args: positional arguments that were passed to the ``__init__``
         method.
        :param kwargs: keyword arguments that were passed to the ``__init__``
         method.

        .. seealso::

            :meth:`.InstanceEvents.init`

            :meth:`.InstanceEvents.load`

        """