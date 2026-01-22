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
def encode_decode_sql(self: Generator, expression: exp.Expression, name: str, replace: bool=True) -> str:
    charset = expression.args.get('charset')
    if charset and charset.name.lower() != 'utf-8':
        self.unsupported(f'Expected utf-8 character set, got {charset}.')
    return self.func(name, expression.this, expression.args.get('replace') if replace else None)