from __future__ import annotations
import sys
import typing as t
from sqlglot import expressions as exp
from sqlglot.dataframe.sql import functions as F
from sqlglot.helper import flatten
def partitionBy(self, *cols: t.Union[ColumnOrName, t.List[ColumnOrName]]) -> WindowSpec:
    from sqlglot.dataframe.sql.column import Column
    cols = flatten(cols) if isinstance(cols[0], (list, set, tuple)) else cols
    expressions = [Column.ensure_col(x).expression for x in cols]
    window_spec = self.copy()
    partition_by_expressions = window_spec.expression.args.get('partition_by', [])
    partition_by_expressions.extend(expressions)
    window_spec.expression.set('partition_by', partition_by_expressions)
    return window_spec