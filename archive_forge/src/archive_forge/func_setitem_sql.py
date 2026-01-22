from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def setitem_sql(self, expression: exp.SetItem) -> str:
    this = expression.this
    if isinstance(this, exp.EQ) and (not isinstance(this.left, exp.Parameter)):
        return f'{self.sql(this.left)} {self.sql(this.right)}'
    return super().setitem_sql(expression)