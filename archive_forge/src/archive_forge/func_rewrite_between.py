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
def rewrite_between(expression: exp.Expression) -> exp.Expression:
    """Rewrite x between y and z to x >= y AND x <= z.

    This is done because comparison simplification is only done on lt/lte/gt/gte.
    """
    if isinstance(expression, exp.Between):
        negate = isinstance(expression.parent, exp.Not)
        expression = exp.and_(exp.GTE(this=expression.this.copy(), expression=expression.args['low']), exp.LTE(this=expression.this.copy(), expression=expression.args['high']), copy=False)
        if negate:
            expression = exp.paren(expression, copy=False)
    return expression