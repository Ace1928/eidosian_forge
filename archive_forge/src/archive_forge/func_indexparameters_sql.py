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
def indexparameters_sql(self, expression: exp.IndexParameters) -> str:
    using = self.sql(expression, 'using')
    using = f' USING {using}' if using else ''
    columns = self.expressions(expression, key='columns', flat=True)
    columns = f'({columns})' if columns else ''
    partition_by = self.expressions(expression, key='partition_by', flat=True)
    partition_by = f' PARTITION BY {partition_by}' if partition_by else ''
    where = self.sql(expression, 'where')
    include = self.expressions(expression, key='include', flat=True)
    if include:
        include = f' INCLUDE ({include})'
    with_storage = self.expressions(expression, key='with_storage', flat=True)
    with_storage = f' WITH ({with_storage})' if with_storage else ''
    tablespace = self.sql(expression, 'tablespace')
    tablespace = f' USING INDEX TABLESPACE {tablespace}' if tablespace else ''
    return f'{using}{columns}{include}{with_storage}{tablespace}{partition_by}{where}'