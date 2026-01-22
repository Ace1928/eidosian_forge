from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.parser import binary_range_parser
from sqlglot.tokens import TokenType
def matchagainst_sql(self, expression: exp.MatchAgainst) -> str:
    this = self.sql(expression, 'this')
    expressions = [f'{self.sql(e)} @@ {this}' for e in expression.expressions]
    sql = ' OR '.join(expressions)
    return f'({sql})' if len(expressions) > 1 else sql