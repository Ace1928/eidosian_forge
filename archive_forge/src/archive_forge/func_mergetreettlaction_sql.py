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
def mergetreettlaction_sql(self, expression: exp.MergeTreeTTLAction) -> str:
    this = self.sql(expression, 'this')
    delete = ' DELETE' if expression.args.get('delete') else ''
    recompress = self.sql(expression, 'recompress')
    recompress = f' RECOMPRESS {recompress}' if recompress else ''
    to_disk = self.sql(expression, 'to_disk')
    to_disk = f' TO DISK {to_disk}' if to_disk else ''
    to_volume = self.sql(expression, 'to_volume')
    to_volume = f' TO VOLUME {to_volume}' if to_volume else ''
    return f'{this}{delete}{recompress}{to_disk}{to_volume}'