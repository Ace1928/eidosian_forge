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
def funcfilter(func: FunctionElement[_T], *criterion: _ColumnExpressionArgument[bool]) -> FunctionFilter[_T]:
    """Produce a :class:`.FunctionFilter` object against a function.

    Used against aggregate and window functions,
    for database backends that support the "FILTER" clause.

    E.g.::

        from sqlalchemy import funcfilter
        funcfilter(func.count(1), MyClass.name == 'some name')

    Would produce "COUNT(1) FILTER (WHERE myclass.name = 'some name')".

    This function is also available from the :data:`~.expression.func`
    construct itself via the :meth:`.FunctionElement.filter` method.

    .. seealso::

        :ref:`tutorial_functions_within_group` - in the
        :ref:`unified_tutorial`

        :meth:`.FunctionElement.filter`

    """
    return FunctionFilter(func, *criterion)