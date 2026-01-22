from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def sort_array(col: ColumnOrName, asc: t.Optional[bool]=None) -> Column:
    if asc is not None:
        return Column.invoke_expression_over_column(col, expression.SortArray, asc=asc)
    return Column.invoke_expression_over_column(col, expression.SortArray)