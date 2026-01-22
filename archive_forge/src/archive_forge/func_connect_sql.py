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
def connect_sql(self, expression: exp.Connect) -> str:
    start = self.sql(expression, 'start')
    start = self.seg(f'START WITH {start}') if start else ''
    nocycle = ' NOCYCLE' if expression.args.get('nocycle') else ''
    connect = self.sql(expression, 'connect')
    connect = self.seg(f'CONNECT BY{nocycle} {connect}')
    return start + connect