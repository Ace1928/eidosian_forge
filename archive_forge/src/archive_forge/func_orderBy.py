from __future__ import annotations
import sys
import typing as t
from sqlglot import expressions as exp
from sqlglot.dataframe.sql import functions as F
from sqlglot.helper import flatten
def orderBy(self, *cols: t.Union[ColumnOrName, t.List[ColumnOrName]]) -> WindowSpec:
    from sqlglot.dataframe.sql.column import Column
    cols = flatten(cols) if isinstance(cols[0], (list, set, tuple)) else cols
    expressions = [Column.ensure_col(x).expression for x in cols]
    window_spec = self.copy()
    if window_spec.expression.args.get('order') is None:
        window_spec.expression.set('order', exp.Order(expressions=[]))
    order_by = window_spec.expression.args['order'].expressions
    order_by.extend(expressions)
    window_spec.expression.args['order'].set('expressions', order_by)
    return window_spec