from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, seq_get
from sqlglot.tokens import TokenType
def join_sql(self, expression: exp.Join) -> str:
    if expression.side == 'LEFT' and (not expression.args.get('on')) and isinstance(expression.this, exp.Unnest):
        return super().join_sql(expression.on(exp.true()))
    return super().join_sql(expression)