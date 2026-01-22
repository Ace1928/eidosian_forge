from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def regexpilike_sql(self, expression: exp.RegexpILike) -> str:
    regex = self.func('CONCAT', "'(?i)'", expression.expression)
    return self.func('match', expression.this, regex)