from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def map_concat(*cols: t.Union[ColumnOrName, t.Iterable[ColumnOrName]]) -> Column:
    columns = list(flatten(cols)) if not isinstance(cols[0], (str, Column)) else cols
    if len(columns) == 1:
        return Column.invoke_anonymous_function(columns[0], 'MAP_CONCAT')
    return Column.invoke_anonymous_function(columns[0], 'MAP_CONCAT', *columns[1:])