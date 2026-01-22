from __future__ import annotations
import datetime
import functools
import itertools
import typing as t
from collections import deque
from decimal import Decimal
from functools import reduce
import sqlglot
from sqlglot import Dialect, exp
from sqlglot.helper import first, merge_ranges, while_changing
from sqlglot.optimizer.scope import find_all_in_scope, walk_in_scope
def simplify_concat(expression):
    """Reduces all groups that contain string literals by concatenating them."""
    if not isinstance(expression, CONCATS) or (isinstance(expression, exp.ConcatWs) and (not expression.expressions[0].is_string)):
        return expression
    if isinstance(expression, exp.ConcatWs):
        sep_expr, *expressions = expression.expressions
        sep = sep_expr.name
        concat_type = exp.ConcatWs
        args = {}
    else:
        expressions = expression.expressions
        sep = ''
        concat_type = exp.Concat
        args = {'safe': expression.args.get('safe'), 'coalesce': expression.args.get('coalesce')}
    new_args = []
    for is_string_group, group in itertools.groupby(expressions or expression.flatten(), lambda e: e.is_string):
        if is_string_group:
            new_args.append(exp.Literal.string(sep.join((string.name for string in group))))
        else:
            new_args.extend(group)
    if len(new_args) == 1 and new_args[0].is_string:
        return new_args[0]
    if concat_type is exp.ConcatWs:
        new_args = [sep_expr] + new_args
    elif isinstance(expression, exp.DPipe):
        return reduce(lambda x, y: exp.DPipe(this=x, expression=y), new_args)
    return concat_type(expressions=new_args, **args)