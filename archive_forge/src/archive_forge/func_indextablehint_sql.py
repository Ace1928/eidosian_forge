from __future__ import annotations
import logging
import re
import typing as t
from collections import defaultdict
from functools import reduce
from sqlglot import exp
from sqlglot.errors import ErrorLevel, UnsupportedError, concat_messages
from sqlglot.helper import apply_index_offset, csv, seq_get
from sqlglot.jsonpath import ALL_JSON_PATH_PARTS, JSON_PATH_PART_TRANSFORMS
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def indextablehint_sql(self, expression: exp.IndexTableHint) -> str:
    this = f'{self.sql(expression, 'this')} INDEX'
    target = self.sql(expression, 'target')
    target = f' FOR {target}' if target else ''
    return f'{this}{target} ({self.expressions(expression, flat=True)})'