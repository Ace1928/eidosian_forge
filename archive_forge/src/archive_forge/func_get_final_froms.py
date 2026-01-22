from __future__ import annotations
import collections
from enum import Enum
import itertools
from typing import AbstractSet
from typing import Any as TODO_Any
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import cache_key
from . import coercions
from . import operators
from . import roles
from . import traversals
from . import type_api
from . import visitors
from ._typing import _ColumnsClauseArgument
from ._typing import _no_kw
from ._typing import _TP
from ._typing import is_column_element
from ._typing import is_select_statement
from ._typing import is_subquery
from ._typing import is_table
from ._typing import is_text_clause
from .annotation import Annotated
from .annotation import SupportsCloneAnnotations
from .base import _clone
from .base import _cloned_difference
from .base import _cloned_intersection
from .base import _entity_namespace_key
from .base import _EntityNamespace
from .base import _expand_cloned
from .base import _from_objects
from .base import _generative
from .base import _never_select_column
from .base import _NoArg
from .base import _select_iterables
from .base import CacheableOptions
from .base import ColumnCollection
from .base import ColumnSet
from .base import CompileState
from .base import DedupeColumnCollection
from .base import Executable
from .base import Generative
from .base import HasCompileState
from .base import HasMemoized
from .base import Immutable
from .coercions import _document_text_coercion
from .elements import _anonymous_label
from .elements import BindParameter
from .elements import BooleanClauseList
from .elements import ClauseElement
from .elements import ClauseList
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import DQLDMLClauseElement
from .elements import GroupedElement
from .elements import literal_column
from .elements import TableValuedColumn
from .elements import UnaryExpression
from .operators import OperatorType
from .sqltypes import NULLTYPE
from .visitors import _TraverseInternalsType
from .visitors import InternalTraversal
from .visitors import prefix_anon_map
from .. import exc
from .. import util
from ..util import HasMemoized_ro_memoized_attribute
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
def get_final_froms(self) -> Sequence[FromClause]:
    """Compute the final displayed list of :class:`_expression.FromClause`
        elements.

        This method will run through the full computation required to
        determine what FROM elements will be displayed in the resulting
        SELECT statement, including shadowing individual tables with
        JOIN objects, as well as full computation for ORM use cases including
        eager loading clauses.

        For ORM use, this accessor returns the **post compilation**
        list of FROM objects; this collection will include elements such as
        eagerly loaded tables and joins.  The objects will **not** be
        ORM enabled and not work as a replacement for the
        :meth:`_sql.Select.select_froms` collection; additionally, the
        method is not well performing for an ORM enabled statement as it
        will incur the full ORM construction process.

        To retrieve the FROM list that's implied by the "columns" collection
        passed to the :class:`_sql.Select` originally, use the
        :attr:`_sql.Select.columns_clause_froms` accessor.

        To select from an alternative set of columns while maintaining the
        FROM list, use the :meth:`_sql.Select.with_only_columns` method and
        pass the
        :paramref:`_sql.Select.with_only_columns.maintain_column_froms`
        parameter.

        .. versionadded:: 1.4.23 - the :meth:`_sql.Select.get_final_froms`
           method replaces the previous :attr:`_sql.Select.froms` accessor,
           which is deprecated.

        .. seealso::

            :attr:`_sql.Select.columns_clause_froms`

        """
    return self._compile_state_factory(self, None)._get_display_froms()