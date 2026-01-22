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
def regexp_replace_sql(self: Generator, expression: exp.RegexpReplace) -> str:
    bad_args = list(filter(expression.args.get, ('position', 'occurrence', 'parameters', 'modifiers')))
    if bad_args:
        self.unsupported(f'REGEXP_REPLACE does not support the following arg(s): {bad_args}')
    return self.func('REGEXP_REPLACE', expression.this, expression.expression, expression.args['replacement'])