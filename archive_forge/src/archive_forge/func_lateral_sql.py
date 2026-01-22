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
def lateral_sql(self, expression: exp.Lateral) -> str:
    this = self.sql(expression, 'this')
    if expression.args.get('view'):
        alias = expression.args['alias']
        columns = self.expressions(alias, key='columns', flat=True)
        table = f' {alias.name}' if alias.name else ''
        columns = f' AS {columns}' if columns else ''
        op_sql = self.seg(f'LATERAL VIEW{(' OUTER' if expression.args.get('outer') else '')}')
        return f'{op_sql}{self.sep()}{this}{table}{columns}'
    alias = self.sql(expression, 'alias')
    alias = f' AS {alias}' if alias else ''
    return f'{self.lateral_op(expression)} {this}{alias}'