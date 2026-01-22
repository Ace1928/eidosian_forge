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
def json_path_key_only_name(self: Generator, expression: exp.JSONPathKey) -> str:
    if isinstance(expression.this, exp.JSONPathWildcard):
        self.unsupported('Unsupported wildcard in JSONPathKey expression')
    return expression.name