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
def checksumproperty_sql(self, expression: exp.ChecksumProperty) -> str:
    if expression.args.get('default'):
        property = 'DEFAULT'
    elif expression.args.get('on'):
        property = 'ON'
    else:
        property = 'OFF'
    return f'CHECKSUM={property}'