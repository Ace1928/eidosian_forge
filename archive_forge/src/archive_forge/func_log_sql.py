from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType
def log_sql(self, expression: exp.Log) -> str:
    if not expression.expression:
        return self.func('LN', expression.this)
    return super().log_sql(expression)