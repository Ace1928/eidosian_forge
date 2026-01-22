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
def bracket_offset_expressions(self, expression: exp.Bracket) -> t.List[exp.Expression]:
    return apply_index_offset(expression.this, expression.expressions, self.dialect.INDEX_OFFSET - expression.args.get('offset', 0))