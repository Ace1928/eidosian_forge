from __future__ import annotations
import contextlib
import typing
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence as typing_Sequence
from typing import Tuple
from . import roles
from .base import _generative
from .base import Executable
from .base import SchemaVisitor
from .elements import ClauseElement
from .. import exc
from .. import util
from ..util import topological
from ..util.typing import Protocol
from ..util.typing import Self
@_generative
def execute_if(self, dialect: Optional[str]=None, callable_: Optional[DDLIfCallable]=None, state: Optional[Any]=None) -> Self:
    """Return a callable that will execute this
        :class:`_ddl.ExecutableDDLElement` conditionally within an event
        handler.

        Used to provide a wrapper for event listening::

            event.listen(
                        metadata,
                        'before_create',
                        DDL("my_ddl").execute_if(dialect='postgresql')
                    )

        :param dialect: May be a string or tuple of strings.
          If a string, it will be compared to the name of the
          executing database dialect::

            DDL('something').execute_if(dialect='postgresql')

          If a tuple, specifies multiple dialect names::

            DDL('something').execute_if(dialect=('postgresql', 'mysql'))

        :param callable\\_: A callable, which will be invoked with
          three positional arguments as well as optional keyword
          arguments:

            :ddl:
              This DDL element.

            :target:
              The :class:`_schema.Table` or :class:`_schema.MetaData`
              object which is the
              target of this event. May be None if the DDL is executed
              explicitly.

            :bind:
              The :class:`_engine.Connection` being used for DDL execution.
              May be None if this construct is being created inline within
              a table, in which case ``compiler`` will be present.

            :tables:
              Optional keyword argument - a list of Table objects which are to
              be created/ dropped within a MetaData.create_all() or drop_all()
              method call.

            :dialect: keyword argument, but always present - the
              :class:`.Dialect` involved in the operation.

            :compiler: keyword argument.  Will be ``None`` for an engine
              level DDL invocation, but will refer to a :class:`.DDLCompiler`
              if this DDL element is being created inline within a table.

            :state:
              Optional keyword argument - will be the ``state`` argument
              passed to this function.

            :checkfirst:
             Keyword argument, will be True if the 'checkfirst' flag was
             set during the call to ``create()``, ``create_all()``,
             ``drop()``, ``drop_all()``.

          If the callable returns a True value, the DDL statement will be
          executed.

        :param state: any value which will be passed to the callable\\_
          as the ``state`` keyword argument.

        .. seealso::

            :meth:`.SchemaItem.ddl_if`

            :class:`.DDLEvents`

            :ref:`event_toplevel`

        """
    self._ddl_if = DDLIf(dialect, callable_, state)
    return self