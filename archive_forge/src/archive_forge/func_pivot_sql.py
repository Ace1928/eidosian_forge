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
def pivot_sql(self, expression: exp.Pivot) -> str:
    expressions = self.expressions(expression, flat=True)
    if expression.this:
        this = self.sql(expression, 'this')
        if not expressions:
            return f'UNPIVOT {this}'
        on = f'{self.seg('ON')} {expressions}'
        using = self.expressions(expression, key='using', flat=True)
        using = f'{self.seg('USING')} {using}' if using else ''
        group = self.sql(expression, 'group')
        return f'PIVOT {this}{on}{using}{group}'
    alias = self.sql(expression, 'alias')
    alias = f' AS {alias}' if alias else ''
    direction = 'UNPIVOT' if expression.unpivot else 'PIVOT'
    field = self.sql(expression, 'field')
    include_nulls = expression.args.get('include_nulls')
    if include_nulls is not None:
        nulls = ' INCLUDE NULLS ' if include_nulls else ' EXCLUDE NULLS '
    else:
        nulls = ''
    return f'{direction}{nulls}({expressions} FOR {field}){alias}'