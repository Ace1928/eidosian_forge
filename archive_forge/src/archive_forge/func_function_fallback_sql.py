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
def function_fallback_sql(self, expression: exp.Func) -> str:
    args = []
    for key in expression.arg_types:
        arg_value = expression.args.get(key)
        if isinstance(arg_value, list):
            for value in arg_value:
                args.append(value)
        elif arg_value is not None:
            args.append(arg_value)
    if self.normalize_functions:
        name = expression.sql_name()
    else:
        name = expression._meta and expression.meta.get('name') or expression.sql_name()
    return self.func(name, *args)