from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def table_parts(self, expression: exp.Table) -> str:
    if expression.meta.get('quoted_table'):
        table_parts = '.'.join((p.name for p in expression.parts))
        return self.sql(exp.Identifier(this=table_parts, quoted=True))
    return super().table_parts(expression)