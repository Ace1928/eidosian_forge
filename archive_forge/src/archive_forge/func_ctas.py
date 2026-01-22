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
def ctas(self, table: ExpOrStr, properties: t.Optional[t.Dict]=None, dialect: DialectType=None, copy: bool=True, **opts) -> Create:
    """
        Convert this expression to a CREATE TABLE AS statement.

        Example:
            >>> Select().select("*").from_("tbl").ctas("x").sql()
            'CREATE TABLE x AS SELECT * FROM tbl'

        Args:
            table: the SQL code string to parse as the table name.
                If another `Expression` instance is passed, it will be used as-is.
            properties: an optional mapping of table properties
            dialect: the dialect used to parse the input table.
            copy: if `False`, modify this expression instance in-place.
            opts: other options to use to parse the input table.

        Returns:
            The new Create expression.
        """
    instance = maybe_copy(self, copy)
    table_expression = maybe_parse(table, into=Table, dialect=dialect, **opts)
    properties_expression = None
    if properties:
        properties_expression = Properties.from_dict(properties)
    return Create(this=table_expression, kind='TABLE', expression=instance, properties=properties_expression)