from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def timetostr_sql(self, expression: exp.TimeToStr) -> str:
    this = expression.this if isinstance(expression.this, exp.TsOrDsToDate) else expression
    return self.func('FORMAT_DATE', self.format_time(expression), this.this)