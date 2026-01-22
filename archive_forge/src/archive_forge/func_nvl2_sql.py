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
def nvl2_sql(self, expression: exp.Nvl2) -> str:
    if self.NVL2_SUPPORTED:
        return self.function_fallback_sql(expression)
    case = exp.Case().when(expression.this.is_(exp.null()).not_(copy=False), expression.args['true'], copy=False)
    else_cond = expression.args.get('false')
    if else_cond:
        case.else_(else_cond, copy=False)
    return self.sql(case)