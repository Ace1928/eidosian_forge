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
def init_collection(self, target: _O, collection: Type[Collection[Any]], collection_adapter: CollectionAdapter) -> None:
    """Receive a 'collection init' event.

        This event is triggered for a collection-based attribute, when
        the initial "empty collection" is first generated for a blank
        attribute, as well as for when the collection is replaced with
        a new one, such as via a set event.

        E.g., given that ``User.addresses`` is a relationship-based
        collection, the event is triggered here::

            u1 = User()
            u1.addresses.append(a1)  #  <- new collection

        and also during replace operations::

            u1.addresses = [a2, a3]  #  <- new collection

        :param target: the object instance receiving the event.
         If the listener is registered with ``raw=True``, this will
         be the :class:`.InstanceState` object.
        :param collection: the new collection.  This will always be generated
         from what was specified as
         :paramref:`_orm.relationship.collection_class`, and will always
         be empty.
        :param collection_adapter: the :class:`.CollectionAdapter` that will
         mediate internal access to the collection.

        .. seealso::

            :class:`.AttributeEvents` - background on listener options such
            as propagation to subclasses.

            :meth:`.AttributeEvents.init_scalar` - "scalar" version of this
            event.

        """