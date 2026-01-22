from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.hive import Hive
from sqlglot.dialects.mysql import MySQL
from sqlglot.helper import apply_index_offset, seq_get
from sqlglot.tokens import TokenType
def strtounix_sql(self, expression: exp.StrToUnix) -> str:
    value_as_text = exp.cast(expression.this, exp.DataType.Type.TEXT)
    parse_without_tz = self.func('DATE_PARSE', value_as_text, self.format_time(expression))
    parse_with_tz = self.func('PARSE_DATETIME', value_as_text, self.format_time(expression, Hive.INVERSE_TIME_MAPPING, Hive.INVERSE_TIME_TRIE))
    coalesced = self.func('COALESCE', self.func('TRY', parse_without_tz), parse_with_tz)
    return self.func('TO_UNIXTIME', coalesced)