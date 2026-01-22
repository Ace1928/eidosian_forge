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
def before_compile_update(self, query: Query[Any], update_context: BulkUpdate) -> None:
    """Allow modifications to the :class:`_query.Query` object within
        :meth:`_query.Query.update`.

        .. deprecated:: 1.4  The :meth:`_orm.QueryEvents.before_compile_update`
           event is superseded by the much more capable
           :meth:`_orm.SessionEvents.do_orm_execute` hook.

        Like the :meth:`.QueryEvents.before_compile` event, if the event
        is to be used to alter the :class:`_query.Query` object, it should
        be configured with ``retval=True``, and the modified
        :class:`_query.Query` object returned, as in ::

            @event.listens_for(Query, "before_compile_update", retval=True)
            def no_deleted(query, update_context):
                for desc in query.column_descriptions:
                    if desc['type'] is User:
                        entity = desc['entity']
                        query = query.filter(entity.deleted == False)

                        update_context.values['timestamp'] = datetime.utcnow()
                return query

        The ``.values`` dictionary of the "update context" object can also
        be modified in place as illustrated above.

        :param query: a :class:`_query.Query` instance; this is also
         the ``.query`` attribute of the given "update context"
         object.

        :param update_context: an "update context" object which is
         the same kind of object as described in
         :paramref:`.QueryEvents.after_bulk_update.update_context`.
         The object has a ``.values`` attribute in an UPDATE context which is
         the dictionary of parameters passed to :meth:`_query.Query.update`.
         This
         dictionary can be modified to alter the VALUES clause of the
         resulting UPDATE statement.

        .. versionadded:: 1.2.17

        .. seealso::

            :meth:`.QueryEvents.before_compile`

            :meth:`.QueryEvents.before_compile_delete`


        """