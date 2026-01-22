from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name, name_sequence
def eliminate_full_outer_join(expression: exp.Expression) -> exp.Expression:
    """
    Converts a query with a FULL OUTER join to a union of identical queries that
    use LEFT/RIGHT OUTER joins instead. This transformation currently only works
    for queries that have a single FULL OUTER join.
    """
    if isinstance(expression, exp.Select):
        full_outer_joins = [(index, join) for index, join in enumerate(expression.args.get('joins') or []) if join.side == 'FULL']
        if len(full_outer_joins) == 1:
            expression_copy = expression.copy()
            expression.set('limit', None)
            index, full_outer_join = full_outer_joins[0]
            full_outer_join.set('side', 'left')
            expression_copy.args['joins'][index].set('side', 'right')
            expression_copy.args.pop('with', None)
            return exp.union(expression, expression_copy, copy=False)
    return expression