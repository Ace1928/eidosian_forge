from __future__ import annotations
import collections.abc as collections_abc
import operator
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import MutableMapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import roles
from . import util as sql_util
from ._typing import _TP
from ._typing import _unexpected_kw
from ._typing import is_column_element
from ._typing import is_named_from_clause
from .base import _entity_namespace_key
from .base import _exclusive_against
from .base import _from_objects
from .base import _generative
from .base import _select_iterables
from .base import ColumnCollection
from .base import CompileState
from .base import DialectKWArgs
from .base import Executable
from .base import Generative
from .base import HasCompileState
from .elements import BooleanClauseList
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import Null
from .selectable import Alias
from .selectable import ExecutableReturnsRows
from .selectable import FromClause
from .selectable import HasCTE
from .selectable import HasPrefixes
from .selectable import Join
from .selectable import SelectLabelStyle
from .selectable import TableClause
from .selectable import TypedReturnsRows
from .sqltypes import NullType
from .visitors import InternalTraversal
from .. import exc
from .. import util
from ..util.typing import Self
from ..util.typing import TypeGuard
@_generative
def ordered_values(self, *args: Tuple[_DMLColumnArgument, Any]) -> Self:
    """Specify the VALUES clause of this UPDATE statement with an explicit
        parameter ordering that will be maintained in the SET clause of the
        resulting UPDATE statement.

        E.g.::

            stmt = table.update().ordered_values(
                ("name", "ed"), ("ident", "foo")
            )

        .. seealso::

           :ref:`tutorial_parameter_ordered_updates` - full example of the
           :meth:`_expression.Update.ordered_values` method.

        .. versionchanged:: 1.4 The :meth:`_expression.Update.ordered_values`
           method
           supersedes the
           :paramref:`_expression.update.preserve_parameter_order`
           parameter, which will be removed in SQLAlchemy 2.0.

        """
    if self._values:
        raise exc.ArgumentError('This statement already has values present')
    elif self._ordered_values:
        raise exc.ArgumentError('This statement already has ordered values present')
    kv_generator = DMLState.get_plugin_class(self)._get_crud_kv_pairs
    self._ordered_values = kv_generator(self, args, True)
    return self