from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def parameterizedagg_sql(self, expression: exp.ParameterizedAgg) -> str:
    params = self.expressions(expression, key='params', flat=True)
    return self.func(expression.name, *expression.expressions) + f'({params})'