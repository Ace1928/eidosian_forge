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
def rowformatdelimitedproperty_sql(self, expression: exp.RowFormatDelimitedProperty) -> str:
    fields = self.sql(expression, 'fields')
    fields = f' FIELDS TERMINATED BY {fields}' if fields else ''
    escaped = self.sql(expression, 'escaped')
    escaped = f' ESCAPED BY {escaped}' if escaped else ''
    items = self.sql(expression, 'collection_items')
    items = f' COLLECTION ITEMS TERMINATED BY {items}' if items else ''
    keys = self.sql(expression, 'map_keys')
    keys = f' MAP KEYS TERMINATED BY {keys}' if keys else ''
    lines = self.sql(expression, 'lines')
    lines = f' LINES TERMINATED BY {lines}' if lines else ''
    null = self.sql(expression, 'null')
    null = f' NULL DEFINED AS {null}' if null else ''
    return f'ROW FORMAT DELIMITED{fields}{escaped}{items}{keys}{lines}{null}'