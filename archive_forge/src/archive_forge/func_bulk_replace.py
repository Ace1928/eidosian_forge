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
def bulk_replace(self, target: _O, values: Iterable[_T], initiator: Event, *, keys: Optional[Iterable[EventConstants]]=None) -> None:
    """Receive a collection 'bulk replace' event.

        This event is invoked for a sequence of values as they are incoming
        to a bulk collection set operation, which can be
        modified in place before the values are treated as ORM objects.
        This is an "early hook" that runs before the bulk replace routine
        attempts to reconcile which objects are already present in the
        collection and which are being removed by the net replace operation.

        It is typical that this method be combined with use of the
        :meth:`.AttributeEvents.append` event.    When using both of these
        events, note that a bulk replace operation will invoke
        the :meth:`.AttributeEvents.append` event for all new items,
        even after :meth:`.AttributeEvents.bulk_replace` has been invoked
        for the collection as a whole.  In order to determine if an
        :meth:`.AttributeEvents.append` event is part of a bulk replace,
        use the symbol :attr:`~.attributes.OP_BULK_REPLACE` to test the
        incoming initiator::

            from sqlalchemy.orm.attributes import OP_BULK_REPLACE

            @event.listens_for(SomeObject.collection, "bulk_replace")
            def process_collection(target, values, initiator):
                values[:] = [_make_value(value) for value in values]

            @event.listens_for(SomeObject.collection, "append", retval=True)
            def process_collection(target, value, initiator):
                # make sure bulk_replace didn't already do it
                if initiator is None or initiator.op is not OP_BULK_REPLACE:
                    return _make_value(value)
                else:
                    return value

        .. versionadded:: 1.2

        :param target: the object instance receiving the event.
          If the listener is registered with ``raw=True``, this will
          be the :class:`.InstanceState` object.
        :param value: a sequence (e.g. a list) of the values being set.  The
          handler can modify this list in place.
        :param initiator: An instance of :class:`.attributes.Event`
          representing the initiation of the event.
        :param keys: When the event is established using the
         :paramref:`.AttributeEvents.include_key` parameter set to
         True, this will be the sequence of keys used in the operation,
         typically only for a dictionary update.  The parameter is not passed
         to the event at all if the the
         :paramref:`.AttributeEvents.include_key`
         was not used to set up the event; this is to allow backwards
         compatibility with existing event handlers that don't include the
         ``key`` parameter.

         .. versionadded:: 2.0

        .. seealso::

            :class:`.AttributeEvents` - background on listener options such
            as propagation to subclasses.


        """