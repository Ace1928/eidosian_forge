from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def version_sql(self, expression: exp.Version) -> str:
    if expression.name == 'TIMESTAMP':
        expression.set('this', 'SYSTEM_TIME')
    return super().version_sql(expression)