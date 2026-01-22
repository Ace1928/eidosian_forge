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
@_lifecycle_event
def loaded_as_persistent(self, session: Session, instance: _O) -> None:
    """Intercept the "loaded as persistent" transition for a specific
        object.

        This event is invoked within the ORM loading process, and is invoked
        very similarly to the :meth:`.InstanceEvents.load` event.  However,
        the event here is linkable to a :class:`.Session` class or instance,
        rather than to a mapper or class hierarchy, and integrates
        with the other session lifecycle events smoothly.  The object
        is guaranteed to be present in the session's identity map when
        this event is called.

        .. note:: This event is invoked within the loader process before
           eager loaders may have been completed, and the object's state may
           not be complete.  Additionally, invoking row-level refresh
           operations on the object will place the object into a new loader
           context, interfering with the existing load context.   See the note
           on :meth:`.InstanceEvents.load` for background on making use of the
           :paramref:`.SessionEvents.restore_load_context` parameter, which
           works in the same manner as that of
           :paramref:`.InstanceEvents.restore_load_context`, in  order to
           resolve this scenario.

        :param session: target :class:`.Session`

        :param instance: the ORM-mapped instance being operated upon.

        .. seealso::

            :ref:`session_lifecycle_events`

        """