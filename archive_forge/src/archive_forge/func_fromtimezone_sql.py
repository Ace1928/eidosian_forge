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
def fromtimezone_sql(self, expression: exp.FromTimeZone) -> str:
    this = self.sql(expression, 'this')
    zone = self.sql(expression, 'zone')
    return f"{this} AT TIME ZONE {zone} AT TIME ZONE 'UTC'"