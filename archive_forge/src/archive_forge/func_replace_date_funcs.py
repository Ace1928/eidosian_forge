from __future__ import annotations
import itertools
import typing as t
from sqlglot import exp
from sqlglot.helper import is_date_unit, is_iso_date, is_iso_datetime
def replace_date_funcs(node: exp.Expression) -> exp.Expression:
    if isinstance(node, (exp.Date, exp.TsOrDsToDate)) and (not node.expressions) and (not node.args.get('zone')):
        return exp.cast(node.this, to=exp.DataType.Type.DATE)
    if isinstance(node, exp.Timestamp) and (not node.expression):
        if not node.type:
            from sqlglot.optimizer.annotate_types import annotate_types
            node = annotate_types(node)
        return exp.cast(node.this, to=node.type or exp.DataType.Type.TIMESTAMP)
    return node