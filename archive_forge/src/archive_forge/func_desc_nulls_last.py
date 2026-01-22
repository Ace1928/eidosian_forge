from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def desc_nulls_last(col: ColumnOrName) -> Column:
    return Column.ensure_col(col).desc_nulls_last()