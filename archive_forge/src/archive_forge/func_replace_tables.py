from __future__ import annotations
import datetime
import math
import numbers
import re
import textwrap
import typing as t
from collections import deque
from copy import deepcopy
from enum import auto
from functools import reduce
from sqlglot.errors import ErrorLevel, ParseError
from sqlglot.helper import (
from sqlglot.tokens import Token
def replace_tables(expression: E, mapping: t.Dict[str, str], dialect: DialectType=None, copy: bool=True) -> E:
    """Replace all tables in expression according to the mapping.

    Args:
        expression: expression node to be transformed and replaced.
        mapping: mapping of table names.
        dialect: the dialect of the mapping table
        copy: whether to copy the expression.

    Examples:
        >>> from sqlglot import exp, parse_one
        >>> replace_tables(parse_one("select * from a.b"), {"a.b": "c"}).sql()
        'SELECT * FROM c /* a.b */'

    Returns:
        The mapped expression.
    """
    mapping = {normalize_table_name(k, dialect=dialect): v for k, v in mapping.items()}

    def _replace_tables(node: Expression) -> Expression:
        if isinstance(node, Table):
            original = normalize_table_name(node, dialect=dialect)
            new_name = mapping.get(original)
            if new_name:
                table = to_table(new_name, **{k: v for k, v in node.args.items() if k not in TABLE_PARTS}, dialect=dialect)
                table.add_comments([original])
                return table
        return node
    return expression.transform(_replace_tables, copy=copy)