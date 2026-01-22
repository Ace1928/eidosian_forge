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
def simplify_coalesce(expression):
    if isinstance(expression, exp.Coalesce) and (not expression.expressions or _is_nonnull_constant(expression.this)) and (not isinstance(expression.parent, exp.Hint)):
        return expression.this
    if not isinstance(expression, COMPARISONS):
        return expression
    if isinstance(expression.left, exp.Coalesce):
        coalesce = expression.left
        other = expression.right
    elif isinstance(expression.right, exp.Coalesce):
        coalesce = expression.right
        other = expression.left
    else:
        return expression
    if not _is_constant(other):
        return expression
    for arg_index, arg in enumerate(coalesce.expressions):
        if _is_constant(arg):
            break
    else:
        return expression
    coalesce.set('expressions', coalesce.expressions[:arg_index])
    coalesce = coalesce if coalesce.expressions else coalesce.this
    return exp.paren(exp.or_(exp.and_(coalesce.is_(exp.null()).not_(copy=False), expression.copy(), copy=False), exp.and_(coalesce.is_(exp.null()), type(expression)(this=arg.copy(), expression=other.copy()), copy=False), copy=False))