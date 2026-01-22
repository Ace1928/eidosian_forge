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
def before_compile_delete(self, query: Query[Any], delete_context: BulkDelete) -> None:
    """Allow modifications to the :class:`_query.Query` object within
        :meth:`_query.Query.delete`.

        .. deprecated:: 1.4  The :meth:`_orm.QueryEvents.before_compile_delete`
           event is superseded by the much more capable
           :meth:`_orm.SessionEvents.do_orm_execute` hook.

        Like the :meth:`.QueryEvents.before_compile` event, this event
        should be configured with ``retval=True``, and the modified
        :class:`_query.Query` object returned, as in ::

            @event.listens_for(Query, "before_compile_delete", retval=True)
            def no_deleted(query, delete_context):
                for desc in query.column_descriptions:
                    if desc['type'] is User:
                        entity = desc['entity']
                        query = query.filter(entity.deleted == False)
                return query

        :param query: a :class:`_query.Query` instance; this is also
         the ``.query`` attribute of the given "delete context"
         object.

        :param delete_context: a "delete context" object which is
         the same kind of object as described in
         :paramref:`.QueryEvents.after_bulk_delete.delete_context`.

        .. versionadded:: 1.2.17

        .. seealso::

            :meth:`.QueryEvents.before_compile`

            :meth:`.QueryEvents.before_compile_update`


        """