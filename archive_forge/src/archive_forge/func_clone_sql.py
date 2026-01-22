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
def clone_sql(self, expression: exp.Clone) -> str:
    this = self.sql(expression, 'this')
    shallow = 'SHALLOW ' if expression.args.get('shallow') else ''
    keyword = 'COPY' if expression.args.get('copy') and self.SUPPORTS_TABLE_COPY else 'CLONE'
    return f'{shallow}{keyword} {this}'