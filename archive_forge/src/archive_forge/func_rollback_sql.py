from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def rollback_sql(self, expression: exp.Rollback) -> str:
    this = self.sql(expression, 'this')
    this = f' {this}' if this else ''
    return f'ROLLBACK TRANSACTION{this}'