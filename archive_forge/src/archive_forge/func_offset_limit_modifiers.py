from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.hive import Hive
from sqlglot.dialects.mysql import MySQL
from sqlglot.helper import apply_index_offset, seq_get
from sqlglot.tokens import TokenType
def offset_limit_modifiers(self, expression: exp.Expression, fetch: bool, limit: t.Optional[exp.Fetch | exp.Limit]) -> t.List[str]:
    return [self.sql(expression, 'offset'), self.sql(limit)]