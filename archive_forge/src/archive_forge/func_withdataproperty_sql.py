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
def withdataproperty_sql(self, expression: exp.WithDataProperty) -> str:
    data_sql = f'WITH {('NO ' if expression.args.get('no') else '')}DATA'
    statistics = expression.args.get('statistics')
    statistics_sql = ''
    if statistics is not None:
        statistics_sql = f' AND {('NO ' if not statistics else '')}STATISTICS'
    return f'{data_sql}{statistics_sql}'