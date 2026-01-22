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
def date_floor(d: datetime.date, unit: str, dialect: Dialect) -> datetime.date:
    if unit == 'year':
        return d.replace(month=1, day=1)
    if unit == 'quarter':
        if d.month <= 3:
            return d.replace(month=1, day=1)
        elif d.month <= 6:
            return d.replace(month=4, day=1)
        elif d.month <= 9:
            return d.replace(month=7, day=1)
        else:
            return d.replace(month=10, day=1)
    if unit == 'month':
        return d.replace(month=d.month, day=1)
    if unit == 'week':
        return d - datetime.timedelta(days=d.weekday() - dialect.WEEK_OFFSET)
    if unit == 'day':
        return d
    raise UnsupportedUnit(f'Unsupported unit: {unit}')