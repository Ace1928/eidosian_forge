from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def column_parts(self, expression: exp.Column) -> str:
    if expression.meta.get('quoted_column'):
        table_parts = '.'.join((p.name for p in expression.parts[:-1]))
        table_path = self.sql(exp.Identifier(this=table_parts, quoted=True))
        return f'{table_path}.{self.sql(expression, 'this')}'
    return super().column_parts(expression)