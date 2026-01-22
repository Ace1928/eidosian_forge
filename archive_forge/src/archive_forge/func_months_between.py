from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def months_between(date1: ColumnOrName, date2: ColumnOrName, roundOff: t.Optional[bool]=None) -> Column:
    if roundOff is None:
        return Column.invoke_expression_over_column(date1, expression.MonthsBetween, expression=date2)
    return Column.invoke_expression_over_column(date1, expression.MonthsBetween, expression=date2, roundoff=roundOff)