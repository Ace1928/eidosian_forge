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
def simplify_literals(expression, root=True):
    if isinstance(expression, exp.Binary) and (not isinstance(expression, exp.Connector)):
        return _flat_simplify(expression, _simplify_binary, root)
    if isinstance(expression, exp.Neg):
        this = expression.this
        if this.is_number:
            value = this.name
            if value[0] == '-':
                return exp.Literal.number(value[1:])
            return exp.Literal.number(f'-{value}')
    if type(expression) in INVERSE_DATE_OPS:
        return _simplify_binary(expression, expression.this, expression.interval()) or expression
    return expression