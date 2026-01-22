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
def jsonobject_sql(self, expression: exp.JSONObject | exp.JSONObjectAgg) -> str:
    null_handling = expression.args.get('null_handling')
    null_handling = f' {null_handling}' if null_handling else ''
    unique_keys = expression.args.get('unique_keys')
    if unique_keys is not None:
        unique_keys = f' {('WITH' if unique_keys else 'WITHOUT')} UNIQUE KEYS'
    else:
        unique_keys = ''
    return_type = self.sql(expression, 'return_type')
    return_type = f' RETURNING {return_type}' if return_type else ''
    encoding = self.sql(expression, 'encoding')
    encoding = f' ENCODING {encoding}' if encoding else ''
    return self.func('JSON_OBJECT' if isinstance(expression, exp.JSONObject) else 'JSON_OBJECTAGG', *expression.expressions, suffix=f'{null_handling}{unique_keys}{return_type}{encoding})')