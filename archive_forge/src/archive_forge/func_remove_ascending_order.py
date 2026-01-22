from __future__ import annotations
import itertools
import typing as t
from sqlglot import exp
from sqlglot.helper import is_date_unit, is_iso_date, is_iso_datetime
def remove_ascending_order(expression: exp.Expression) -> exp.Expression:
    if isinstance(expression, exp.Ordered) and expression.args.get('desc') is False:
        expression.set('desc', None)
    return expression