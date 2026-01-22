from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.tokens import TokenType
def rangen_sql(self, expression: exp.RangeN) -> str:
    this = self.sql(expression, 'this')
    expressions_sql = self.expressions(expression)
    each_sql = self.sql(expression, 'each')
    each_sql = f' EACH {each_sql}' if each_sql else ''
    return f'RANGE_N({this} BETWEEN {expressions_sql}{each_sql})'