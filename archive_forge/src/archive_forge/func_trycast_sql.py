from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def trycast_sql(self, expression: exp.TryCast) -> str:
    return self.cast_sql(expression, safe_prefix='SAFE_')