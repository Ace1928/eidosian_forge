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
def remove_where_true(expression):
    for where in expression.find_all(exp.Where):
        if always_true(where.this):
            where.pop()
    for join in expression.find_all(exp.Join):
        if always_true(join.args.get('on')) and (not join.args.get('using')) and (not join.args.get('method')) and ((join.side, join.kind) in JOINS):
            join.args['on'].pop()
            join.set('side', None)
            join.set('kind', 'CROSS')