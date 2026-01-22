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
def lock_sql(self, expression: exp.Lock) -> str:
    if not self.LOCKING_READS_SUPPORTED:
        self.unsupported("Locking reads using 'FOR UPDATE/SHARE' are not supported")
        return ''
    lock_type = 'FOR UPDATE' if expression.args['update'] else 'FOR SHARE'
    expressions = self.expressions(expression, flat=True)
    expressions = f' OF {expressions}' if expressions else ''
    wait = expression.args.get('wait')
    if wait is not None:
        if isinstance(wait, exp.Literal):
            wait = f' WAIT {self.sql(wait)}'
        else:
            wait = ' NOWAIT' if wait else ' SKIP LOCKED'
    return f'{lock_type}{expressions}{wait or ''}'