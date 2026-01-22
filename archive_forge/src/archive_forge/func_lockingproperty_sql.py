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
def lockingproperty_sql(self, expression: exp.LockingProperty) -> str:
    kind = expression.args.get('kind')
    this = f' {self.sql(expression, 'this')}' if expression.this else ''
    for_or_in = expression.args.get('for_or_in')
    for_or_in = f' {for_or_in}' if for_or_in else ''
    lock_type = expression.args.get('lock_type')
    override = ' OVERRIDE' if expression.args.get('override') else ''
    return f'LOCKING {kind}{this}{for_or_in} {lock_type}{override}'