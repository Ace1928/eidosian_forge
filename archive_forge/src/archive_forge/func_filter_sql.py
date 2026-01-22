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
def filter_sql(self, expression: exp.Filter) -> str:
    if self.AGGREGATE_FILTER_SUPPORTED:
        this = self.sql(expression, 'this')
        where = self.sql(expression, 'expression').strip()
        return f'{this} FILTER({where})'
    agg = expression.this
    agg_arg = agg.this
    cond = expression.expression.this
    agg_arg.replace(exp.If(this=cond.copy(), true=agg_arg.copy()))
    return self.sql(agg)