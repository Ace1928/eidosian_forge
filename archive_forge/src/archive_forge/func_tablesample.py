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
def tablesample(selectable: _FromClauseArgument, sampling: Union[float, Function[Any]], name: Optional[str]=None, seed: Optional[roles.ExpressionElementRole[Any]]=None) -> TableSample:
    """Return a :class:`_expression.TableSample` object.

    :class:`_expression.TableSample` is an :class:`_expression.Alias`
    subclass that represents
    a table with the TABLESAMPLE clause applied to it.
    :func:`_expression.tablesample`
    is also available from the :class:`_expression.FromClause`
    class via the
    :meth:`_expression.FromClause.tablesample` method.

    The TABLESAMPLE clause allows selecting a randomly selected approximate
    percentage of rows from a table. It supports multiple sampling methods,
    most commonly BERNOULLI and SYSTEM.

    e.g.::

        from sqlalchemy import func

        selectable = people.tablesample(
                    func.bernoulli(1),
                    name='alias',
                    seed=func.random())
        stmt = select(selectable.c.people_id)

    Assuming ``people`` with a column ``people_id``, the above
    statement would render as::

        SELECT alias.people_id FROM
        people AS alias TABLESAMPLE bernoulli(:bernoulli_1)
        REPEATABLE (random())

    :param sampling: a ``float`` percentage between 0 and 100 or
        :class:`_functions.Function`.

    :param name: optional alias name

    :param seed: any real-valued SQL expression.  When specified, the
     REPEATABLE sub-clause is also rendered.

    """
    return TableSample._factory(selectable, sampling, name=name, seed=seed)