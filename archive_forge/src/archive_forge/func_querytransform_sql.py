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
def querytransform_sql(self, expression: exp.QueryTransform) -> str:
    transform = self.func('TRANSFORM', *expression.expressions)
    row_format_before = self.sql(expression, 'row_format_before')
    row_format_before = f' {row_format_before}' if row_format_before else ''
    record_writer = self.sql(expression, 'record_writer')
    record_writer = f' RECORDWRITER {record_writer}' if record_writer else ''
    using = f' USING {self.sql(expression, 'command_script')}'
    schema = self.sql(expression, 'schema')
    schema = f' AS {schema}' if schema else ''
    row_format_after = self.sql(expression, 'row_format_after')
    row_format_after = f' {row_format_after}' if row_format_after else ''
    record_reader = self.sql(expression, 'record_reader')
    record_reader = f' RECORDREADER {record_reader}' if record_reader else ''
    return f'{transform}{row_format_before}{record_writer}{using}{schema}{row_format_after}{record_reader}'