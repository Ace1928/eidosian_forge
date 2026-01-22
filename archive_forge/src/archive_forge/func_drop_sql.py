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
def drop_sql(self, expression: exp.Drop) -> str:
    this = self.sql(expression, 'this')
    expressions = self.expressions(expression, flat=True)
    expressions = f' ({expressions})' if expressions else ''
    kind = expression.args['kind']
    exists_sql = ' IF EXISTS ' if expression.args.get('exists') else ' '
    temporary = ' TEMPORARY' if expression.args.get('temporary') else ''
    materialized = ' MATERIALIZED' if expression.args.get('materialized') else ''
    cascade = ' CASCADE' if expression.args.get('cascade') else ''
    constraints = ' CONSTRAINTS' if expression.args.get('constraints') else ''
    purge = ' PURGE' if expression.args.get('purge') else ''
    return f'DROP{temporary}{materialized} {kind}{exists_sql}{this}{expressions}{cascade}{constraints}{purge}'