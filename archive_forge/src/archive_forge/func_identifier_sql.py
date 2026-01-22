from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def identifier_sql(self, expression: exp.Identifier) -> str:
    identifier = super().identifier_sql(expression)
    if expression.args.get('global'):
        identifier = f'##{identifier}'
    elif expression.args.get('temporary'):
        identifier = f'#{identifier}'
    return identifier