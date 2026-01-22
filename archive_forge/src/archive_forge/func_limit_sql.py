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
def limit_sql(self, expression: exp.Limit, top: bool=False) -> str:
    this = self.sql(expression, 'this')
    args = [self._simplify_unless_literal(e) if self.LIMIT_ONLY_LITERALS else e for e in (expression.args.get(k) for k in ('offset', 'expression')) if e]
    args_sql = ', '.join((self.sql(e) for e in args))
    args_sql = f'({args_sql})' if top and any((not e.is_number for e in args)) else args_sql
    expressions = self.expressions(expression, flat=True)
    expressions = f' BY {expressions}' if expressions else ''
    return f'{this}{self.seg('TOP' if top else 'LIMIT')} {args_sql}{expressions}'