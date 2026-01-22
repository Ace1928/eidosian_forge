from __future__ import annotations
from typing import Any
from typing import Optional
from typing import overload
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import roles
from ._typing import _ColumnsClauseArgument
from ._typing import _no_kw
from .elements import ColumnClause
from .selectable import Alias
from .selectable import CompoundSelect
from .selectable import Exists
from .selectable import FromClause
from .selectable import Join
from .selectable import Lateral
from .selectable import LateralFromClause
from .selectable import NamedFromClause
from .selectable import Select
from .selectable import TableClause
from .selectable import TableSample
from .selectable import Values
def lateral(selectable: Union[SelectBase, _FromClauseArgument], name: Optional[str]=None) -> LateralFromClause:
    """Return a :class:`_expression.Lateral` object.

    :class:`_expression.Lateral` is an :class:`_expression.Alias`
    subclass that represents
    a subquery with the LATERAL keyword applied to it.

    The special behavior of a LATERAL subquery is that it appears in the
    FROM clause of an enclosing SELECT, but may correlate to other
    FROM clauses of that SELECT.   It is a special case of subquery
    only supported by a small number of backends, currently more recent
    PostgreSQL versions.

    .. seealso::

        :ref:`tutorial_lateral_correlation` -  overview of usage.

    """
    return Lateral._factory(selectable, name=name)