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
def simplify_connectors(expression, root=True):

    def _simplify_connectors(expression, left, right):
        if left == right:
            return left
        if isinstance(expression, exp.And):
            if is_false(left) or is_false(right):
                return exp.false()
            if is_null(left) or is_null(right):
                return exp.null()
            if always_true(left) and always_true(right):
                return exp.true()
            if always_true(left):
                return right
            if always_true(right):
                return left
            return _simplify_comparison(expression, left, right)
        elif isinstance(expression, exp.Or):
            if always_true(left) or always_true(right):
                return exp.true()
            if is_false(left) and is_false(right):
                return exp.false()
            if is_null(left) and is_null(right) or (is_null(left) and is_false(right)) or (is_false(left) and is_null(right)):
                return exp.null()
            if is_false(left):
                return right
            if is_false(right):
                return left
            return _simplify_comparison(expression, left, right, or_=True)
    if isinstance(expression, exp.Connector):
        return _flat_simplify(expression, _simplify_connectors, root)
    return expression