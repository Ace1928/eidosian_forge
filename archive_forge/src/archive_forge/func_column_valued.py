from __future__ import annotations
import datetime
import decimal
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import annotation
from . import coercions
from . import operators
from . import roles
from . import schema
from . import sqltypes
from . import type_api
from . import util as sqlutil
from ._typing import is_table_value_type
from .base import _entity_namespace
from .base import ColumnCollection
from .base import Executable
from .base import Generative
from .base import HasMemoized
from .elements import _type_from_args
from .elements import BinaryExpression
from .elements import BindParameter
from .elements import Cast
from .elements import ClauseList
from .elements import ColumnElement
from .elements import Extract
from .elements import FunctionFilter
from .elements import Grouping
from .elements import literal_column
from .elements import NamedColumn
from .elements import Over
from .elements import WithinGroup
from .selectable import FromClause
from .selectable import Select
from .selectable import TableValuedAlias
from .sqltypes import TableValueType
from .type_api import TypeEngine
from .visitors import InternalTraversal
from .. import util
def column_valued(self, name: Optional[str]=None, joins_implicitly: bool=False) -> TableValuedColumn[_T]:
    """Return this :class:`_functions.FunctionElement` as a column expression that
        selects from itself as a FROM clause.

        E.g.:

        .. sourcecode:: pycon+sql

            >>> from sqlalchemy import select, func
            >>> gs = func.generate_series(1, 5, -1).column_valued()
            >>> print(select(gs))
            {printsql}SELECT anon_1
            FROM generate_series(:generate_series_1, :generate_series_2, :generate_series_3) AS anon_1

        This is shorthand for::

            gs = func.generate_series(1, 5, -1).alias().column

        :param name: optional name to assign to the alias name that's generated.
         If omitted, a unique anonymizing name is used.

        :param joins_implicitly: when True, the "table" portion of the column
         valued function may be a member of the FROM clause without any
         explicit JOIN to other tables in the SQL query, and no "cartesian
         product" warning will be generated. May be useful for SQL functions
         such as ``func.json_array_elements()``.

         .. versionadded:: 1.4.46

        .. seealso::

            :ref:`tutorial_functions_column_valued` - in the :ref:`unified_tutorial`

            :ref:`postgresql_column_valued` - in the :ref:`postgresql_toplevel` documentation

            :meth:`_functions.FunctionElement.table_valued`

        """
    return self.alias(name=name, joins_implicitly=joins_implicitly).column