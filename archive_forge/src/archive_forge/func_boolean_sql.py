from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def boolean_sql(self, expression: exp.Boolean) -> str:
    if type(expression.parent) in BIT_TYPES:
        return '1' if expression.this else '0'
    return '(1 = 1)' if expression.this else '(1 = 0)'