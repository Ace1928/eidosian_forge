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
def union_op(self, expression: exp.Union) -> str:
    kind = ' DISTINCT' if self.EXPLICIT_UNION else ''
    kind = kind if expression.args.get('distinct') else ' ALL'
    by_name = ' BY NAME' if expression.args.get('by_name') else ''
    return f'UNION{kind}{by_name}'