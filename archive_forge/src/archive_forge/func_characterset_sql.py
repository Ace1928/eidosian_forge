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
def characterset_sql(self, expression: exp.CharacterSet) -> str:
    if isinstance(expression.parent, exp.Cast):
        return f'CHAR CHARACTER SET {self.sql(expression, 'this')}'
    default = 'DEFAULT ' if expression.args.get('default') else ''
    return f'{default}CHARACTER SET={self.sql(expression, 'this')}'