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
def tochar_sql(self, expression: exp.ToChar) -> str:
    if expression.args.get('format'):
        self.unsupported('Format argument unsupported for TO_CHAR/TO_VARCHAR function')
    return self.sql(exp.cast(expression.this, exp.DataType.Type.TEXT))