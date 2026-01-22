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
def columnposition_sql(self, expression: exp.ColumnPosition) -> str:
    this = self.sql(expression, 'this')
    this = f' {this}' if this else ''
    position = self.sql(expression, 'position')
    return f'{position}{this}'