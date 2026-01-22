from __future__ import annotations
import logging
import typing as t
from enum import Enum, auto
from functools import reduce
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.generator import Generator
from sqlglot.helper import AutoName, flatten, is_int, seq_get
from sqlglot.jsonpath import parse as parse_json_path
from sqlglot.parser import Parser
from sqlglot.time import TIMEZONES, format_time
from sqlglot.tokens import Token, Tokenizer, TokenType
from sqlglot.trie import new_trie
def no_last_day_sql(self: Generator, expression: exp.LastDay) -> str:
    trunc_curr_date = exp.func('date_trunc', 'month', expression.this)
    plus_one_month = exp.func('date_add', trunc_curr_date, 1, 'month')
    minus_one_day = exp.func('date_sub', plus_one_month, 1, 'day')
    return self.sql(exp.cast(minus_one_day, exp.DataType.Type.DATE))