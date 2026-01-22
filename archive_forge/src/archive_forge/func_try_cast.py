from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple as typing_Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import roles
from .base import _NoArg
from .coercions import _document_text_coercion
from .elements import BindParameter
from .elements import BooleanClauseList
from .elements import Case
from .elements import Cast
from .elements import CollationClause
from .elements import CollectionAggregate
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import Extract
from .elements import False_
from .elements import FunctionFilter
from .elements import Label
from .elements import Null
from .elements import Over
from .elements import TextClause
from .elements import True_
from .elements import TryCast
from .elements import Tuple
from .elements import TypeCoerce
from .elements import UnaryExpression
from .elements import WithinGroup
from .functions import FunctionElement
from ..util.typing import Literal
def try_cast(expression: _ColumnExpressionOrLiteralArgument[Any], type_: _TypeEngineArgument[_T]) -> TryCast[_T]:
    """Produce a ``TRY_CAST`` expression for backends which support it;
    this is a ``CAST`` which returns NULL for un-castable conversions.

    In SQLAlchemy, this construct is supported **only** by the SQL Server
    dialect, and will raise a :class:`.CompileError` if used on other
    included backends.  However, third party backends may also support
    this construct.

    .. tip:: As :func:`_sql.try_cast` originates from the SQL Server dialect,
       it's importable both from ``sqlalchemy.`` as well as from
       ``sqlalchemy.dialects.mssql``.

    :func:`_sql.try_cast` returns an instance of :class:`.TryCast` and
    generally behaves similarly to the :class:`.Cast` construct;
    at the SQL level, the difference between ``CAST`` and ``TRY_CAST``
    is that ``TRY_CAST`` returns NULL for an un-castable expression,
    such as attempting to cast a string ``"hi"`` to an integer value.

    E.g.::

        from sqlalchemy import select, try_cast, Numeric

        stmt = select(
            try_cast(product_table.c.unit_price, Numeric(10, 4))
        )

    The above would render on Microsoft SQL Server as::

        SELECT TRY_CAST (product_table.unit_price AS NUMERIC(10, 4))
        FROM product_table

    .. versionadded:: 2.0.14  :func:`.try_cast` has been
       generalized from the SQL Server dialect into a general use
       construct that may be supported by additional dialects.

    """
    return TryCast(expression, type_)