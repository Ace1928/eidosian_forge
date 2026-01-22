from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def eq_sql(self, expression: exp.EQ) -> str:
    if isinstance(expression.left, exp.Null) or isinstance(expression.right, exp.Null):
        if not isinstance(expression.parent, exp.Update):
            return 'NULL'
    return self.binary(expression, '=')