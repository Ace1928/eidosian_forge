from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.tokens import TokenType
def update_sql(self, expression: exp.Update) -> str:
    this = self.sql(expression, 'this')
    from_sql = self.sql(expression, 'from')
    set_sql = self.expressions(expression, flat=True)
    where_sql = self.sql(expression, 'where')
    sql = f'UPDATE {this}{from_sql} SET {set_sql}{where_sql}'
    return self.prepend_ctes(expression, sql)