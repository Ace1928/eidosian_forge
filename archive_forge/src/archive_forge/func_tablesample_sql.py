from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, seq_get
from sqlglot.tokens import TokenType
def tablesample_sql(self, expression: exp.TableSample, sep: str=' AS ', tablesample_keyword: t.Optional[str]=None) -> str:
    if not isinstance(expression.parent, exp.Select):
        tablesample_keyword = 'TABLESAMPLE'
    return super().tablesample_sql(expression, sep=sep, tablesample_keyword=tablesample_keyword)