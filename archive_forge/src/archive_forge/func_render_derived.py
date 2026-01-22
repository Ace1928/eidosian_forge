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
def render_derived(self, name: Optional[str]=None, with_types: bool=False) -> TableValuedAlias:
    """Apply "render derived" to this :class:`_sql.TableValuedAlias`.

        This has the effect of the individual column names listed out
        after the alias name in the "AS" sequence, e.g.:

        .. sourcecode:: pycon+sql

            >>> print(
            ...     select(
            ...         func.unnest(array(["one", "two", "three"])).
                        table_valued("x", with_ordinality="o").render_derived()
            ...     )
            ... )
            {printsql}SELECT anon_1.x, anon_1.o
            FROM unnest(ARRAY[%(param_1)s, %(param_2)s, %(param_3)s]) WITH ORDINALITY AS anon_1(x, o)

        The ``with_types`` keyword will render column types inline within
        the alias expression (this syntax currently applies to the
        PostgreSQL database):

        .. sourcecode:: pycon+sql

            >>> print(
            ...     select(
            ...         func.json_to_recordset(
            ...             '[{"a":1,"b":"foo"},{"a":"2","c":"bar"}]'
            ...         )
            ...         .table_valued(column("a", Integer), column("b", String))
            ...         .render_derived(with_types=True)
            ...     )
            ... )
            {printsql}SELECT anon_1.a, anon_1.b FROM json_to_recordset(:json_to_recordset_1)
            AS anon_1(a INTEGER, b VARCHAR)

        :param name: optional string name that will be applied to the alias
         generated.  If left as None, a unique anonymizing name will be used.

        :param with_types: if True, the derived columns will include the
         datatype specification with each column. This is a special syntax
         currently known to be required by PostgreSQL for some SQL functions.

        """
    new_alias: TableValuedAlias = TableValuedAlias._construct(self.element, name=name, table_value_type=self._tableval_type, joins_implicitly=self.joins_implicitly)
    new_alias._render_derived = True
    new_alias._render_derived_w_types = with_types
    return new_alias