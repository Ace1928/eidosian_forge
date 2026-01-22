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
def detached_to_persistent(self, session: Session, instance: _O) -> None:
    """Intercept the "detached to persistent" transition for a specific
        object.

        This event is a specialization of the
        :meth:`.SessionEvents.after_attach` event which is only invoked
        for this specific transition.  It is invoked typically during the
        :meth:`.Session.add` call, as well as during the
        :meth:`.Session.delete` call if the object was not previously
        associated with the
        :class:`.Session` (note that an object marked as "deleted" remains
        in the "persistent" state until the flush proceeds).

        .. note::

            If the object becomes persistent as part of a call to
            :meth:`.Session.delete`, the object is **not** yet marked as
            deleted when this event is called.  To detect deleted objects,
            check the ``deleted`` flag sent to the
            :meth:`.SessionEvents.persistent_to_detached` to event after the
            flush proceeds, or check the :attr:`.Session.deleted` collection
            within the :meth:`.SessionEvents.before_flush` event if deleted
            objects need to be intercepted before the flush.

        :param session: target :class:`.Session`

        :param instance: the ORM-mapped instance being operated upon.

        .. seealso::

            :ref:`session_lifecycle_events`

        """