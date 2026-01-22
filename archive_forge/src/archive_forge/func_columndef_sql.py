from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, seq_get
from sqlglot.tokens import TokenType
def columndef_sql(self, expression: exp.ColumnDef, sep: str=' ') -> str:
    if isinstance(expression.parent, exp.UserDefinedFunction):
        return self.sql(expression, 'this')
    return super().columndef_sql(expression, sep)