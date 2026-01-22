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
def simplify_startswith(expression: exp.Expression) -> exp.Expression:
    """
    Reduces a prefix check to either TRUE or FALSE if both the string and the
    prefix are statically known.

    Example:
        >>> from sqlglot import parse_one
        >>> simplify_startswith(parse_one("STARTSWITH('foo', 'f')")).sql()
        'TRUE'
    """
    if isinstance(expression, exp.StartsWith) and expression.this.is_string and expression.expression.is_string:
        return exp.convert(expression.name.startswith(expression.expression.name))
    return expression