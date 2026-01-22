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
def predict_sql(self, expression: exp.Predict) -> str:
    model = self.sql(expression, 'this')
    model = f'MODEL {model}'
    table = self.sql(expression, 'expression')
    table = f'TABLE {table}' if not isinstance(expression.expression, exp.Subquery) else table
    parameters = self.sql(expression, 'params_struct')
    return self.func('PREDICT', model, table, parameters or None)