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
def dispose_collection(self, target: _O, collection: Collection[Any], collection_adapter: CollectionAdapter) -> None:
    """Receive a 'collection dispose' event.

        This event is triggered for a collection-based attribute when
        a collection is replaced, that is::

            u1.addresses.append(a1)

            u1.addresses = [a2, a3]  # <- old collection is disposed

        The old collection received will contain its previous contents.

        .. versionchanged:: 1.2 The collection passed to
           :meth:`.AttributeEvents.dispose_collection` will now have its
           contents before the dispose intact; previously, the collection
           would be empty.

        .. seealso::

            :class:`.AttributeEvents` - background on listener options such
            as propagation to subclasses.

        """