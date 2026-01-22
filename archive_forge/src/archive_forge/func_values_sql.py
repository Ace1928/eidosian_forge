from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType
def values_sql(self, expression: exp.Values, values_as_table: bool=True) -> str:
    if expression.find(*self.UNSUPPORTED_VALUES_EXPRESSIONS):
        values_as_table = False
    return super().values_sql(expression, values_as_table=values_as_table)