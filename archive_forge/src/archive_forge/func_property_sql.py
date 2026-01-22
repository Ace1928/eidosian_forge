from __future__ import annotations
from sqlglot import exp
from sqlglot.dialects.trino import Trino
from sqlglot.tokens import TokenType
def property_sql(self, expression: exp.Property) -> str:
    return f'{self.property_name(expression, string_key=True)}={self.sql(expression, 'value')}'