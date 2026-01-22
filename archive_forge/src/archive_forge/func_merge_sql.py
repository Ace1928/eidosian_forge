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
def merge_sql(self, expression: exp.Merge) -> str:
    table = expression.this
    table_alias = ''
    hints = table.args.get('hints')
    if hints and table.alias and isinstance(hints[0], exp.WithTableHint):
        table_alias = f' AS {self.sql(table.args['alias'].pop())}'
    this = self.sql(table)
    using = f'USING {self.sql(expression, 'using')}'
    on = f'ON {self.sql(expression, 'on')}'
    expressions = self.expressions(expression, sep=' ')
    return self.prepend_ctes(expression, f'MERGE INTO {this}{table_alias} {using} {on} {expressions}')