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
def before_mapper_configured(self, mapper: Mapper[_O], class_: Type[_O]) -> None:
    """Called right before a specific mapper is to be configured.

        This event is intended to allow a specific mapper to be skipped during
        the configure step, by returning the :attr:`.orm.interfaces.EXT_SKIP`
        symbol which indicates to the :func:`.configure_mappers` call that this
        particular mapper (or hierarchy of mappers, if ``propagate=True`` is
        used) should be skipped in the current configuration run. When one or
        more mappers are skipped, the he "new mappers" flag will remain set,
        meaning the :func:`.configure_mappers` function will continue to be
        called when mappers are used, to continue to try to configure all
        available mappers.

        In comparison to the other configure-level events,
        :meth:`.MapperEvents.before_configured`,
        :meth:`.MapperEvents.after_configured`, and
        :meth:`.MapperEvents.mapper_configured`, the
        :meth;`.MapperEvents.before_mapper_configured` event provides for a
        meaningful return value when it is registered with the ``retval=True``
        parameter.

        .. versionadded:: 1.3

        e.g.::

            from sqlalchemy.orm import EXT_SKIP

            Base = declarative_base()

            DontConfigureBase = declarative_base()

            @event.listens_for(
                DontConfigureBase,
                "before_mapper_configured", retval=True, propagate=True)
            def dont_configure(mapper, cls):
                return EXT_SKIP


        .. seealso::

            :meth:`.MapperEvents.before_configured`

            :meth:`.MapperEvents.after_configured`

            :meth:`.MapperEvents.mapper_configured`

        """