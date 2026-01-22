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
def mergeblockratioproperty_sql(self, expression: exp.MergeBlockRatioProperty) -> str:
    if expression.args.get('no'):
        return 'NO MERGEBLOCKRATIO'
    if expression.args.get('default'):
        return 'DEFAULT MERGEBLOCKRATIO'
    percent = ' PERCENT' if expression.args.get('percent') else ''
    return f'MERGEBLOCKRATIO={self.sql(expression, 'this')}{percent}'