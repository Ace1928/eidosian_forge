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
def inputoutputformat_sql(self, expression: exp.InputOutputFormat) -> str:
    input_format = self.sql(expression, 'input_format')
    input_format = f'INPUTFORMAT {input_format}' if input_format else ''
    output_format = self.sql(expression, 'output_format')
    output_format = f'OUTPUTFORMAT {output_format}' if output_format else ''
    return self.sep().join((input_format, output_format))