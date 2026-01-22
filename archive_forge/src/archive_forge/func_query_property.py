from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .session import _S
from .session import Session
from .. import exc as sa_exc
from .. import util
from ..util import create_proxy_methods
from ..util import ScopedRegistry
from ..util import ThreadLocalRegistry
from ..util import warn
from ..util import warn_deprecated
from ..util.typing import Protocol
def query_property(self, query_cls: Optional[Type[Query[_T]]]=None) -> QueryPropertyDescriptor:
    """return a class property which produces a legacy
        :class:`_query.Query` object against the class and the current
        :class:`.Session` when called.

        .. legacy:: The :meth:`_orm.scoped_session.query_property` accessor
           is specific to the legacy :class:`.Query` object and is not
           considered to be part of :term:`2.0-style` ORM use.

        e.g.::

            from sqlalchemy.orm import QueryPropertyDescriptor
            from sqlalchemy.orm import scoped_session
            from sqlalchemy.orm import sessionmaker

            Session = scoped_session(sessionmaker())

            class MyClass:
                query: QueryPropertyDescriptor = Session.query_property()

            # after mappers are defined
            result = MyClass.query.filter(MyClass.name=='foo').all()

        Produces instances of the session's configured query class by
        default.  To override and use a custom implementation, provide
        a ``query_cls`` callable.  The callable will be invoked with
        the class's mapper as a positional argument and a session
        keyword argument.

        There is no limit to the number of query properties placed on
        a class.

        """

    class query:

        def __get__(s, instance: Any, owner: Type[_O]) -> Query[_O]:
            if query_cls:
                return query_cls(owner, session=self.registry())
            else:
                return self.registry().query(owner)
    return query()