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
def indexconstraintoption_sql(self, expression: exp.IndexConstraintOption) -> str:
    key_block_size = self.sql(expression, 'key_block_size')
    if key_block_size:
        return f'KEY_BLOCK_SIZE = {key_block_size}'
    using = self.sql(expression, 'using')
    if using:
        return f'USING {using}'
    parser = self.sql(expression, 'parser')
    if parser:
        return f'WITH PARSER {parser}'
    comment = self.sql(expression, 'comment')
    if comment:
        return f'COMMENT {comment}'
    visible = expression.args.get('visible')
    if visible is not None:
        return 'VISIBLE' if visible else 'INVISIBLE'
    engine_attr = self.sql(expression, 'engine_attr')
    if engine_attr:
        return f'ENGINE_ATTRIBUTE = {engine_attr}'
    secondary_engine_attr = self.sql(expression, 'secondary_engine_attr')
    if secondary_engine_attr:
        return f'SECONDARY_ENGINE_ATTRIBUTE = {secondary_engine_attr}'
    self.unsupported('Unsupported index constraint option.')
    return ''