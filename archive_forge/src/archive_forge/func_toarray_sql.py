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
def toarray_sql(self, expression: exp.ToArray) -> str:
    arg = expression.this
    if not arg.type:
        from sqlglot.optimizer.annotate_types import annotate_types
        arg = annotate_types(arg)
    if arg.is_type(exp.DataType.Type.ARRAY):
        return self.sql(arg)
    cond_for_null = arg.is_(exp.null())
    return self.sql(exp.func('IF', cond_for_null, exp.null(), exp.array(arg, copy=False)))