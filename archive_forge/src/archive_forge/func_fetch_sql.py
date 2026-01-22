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
def fetch_sql(self, expression: exp.Fetch) -> str:
    direction = expression.args.get('direction')
    direction = f' {direction}' if direction else ''
    count = expression.args.get('count')
    count = f' {count}' if count else ''
    if expression.args.get('percent'):
        count = f'{count} PERCENT'
    with_ties_or_only = 'WITH TIES' if expression.args.get('with_ties') else 'ONLY'
    return f'{self.seg('FETCH')}{direction}{count} ROWS {with_ties_or_only}'