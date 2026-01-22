from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def bround(col: ColumnOrName, scale: t.Optional[int]=None) -> Column:
    if scale is not None:
        return Column.invoke_anonymous_function(col, 'BROUND', scale)
    return Column.invoke_anonymous_function(col, 'BROUND')