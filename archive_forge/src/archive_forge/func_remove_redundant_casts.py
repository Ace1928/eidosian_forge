from __future__ import annotations
import itertools
import typing as t
from sqlglot import exp
from sqlglot.helper import is_date_unit, is_iso_date, is_iso_datetime
def remove_redundant_casts(expression: exp.Expression) -> exp.Expression:
    if isinstance(expression, exp.Cast) and expression.this.type and (expression.to.this == expression.this.type.this):
        return expression.this
    return expression