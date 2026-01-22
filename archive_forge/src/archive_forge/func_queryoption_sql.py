from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def queryoption_sql(self, expression: exp.QueryOption) -> str:
    option = self.sql(expression, 'this')
    value = self.sql(expression, 'expression')
    if value:
        optional_equal_sign = '= ' if option in OPTIONS_THAT_REQUIRE_EQUAL else ''
        return f'{option} {optional_equal_sign}{value}'
    return option