from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def date_add(col: ColumnOrName, days: t.Union[ColumnOrName, int]) -> Column:
    return Column.invoke_expression_over_column(col, expression.DateAdd, expression=days, unit=expression.Var(this='DAY'))