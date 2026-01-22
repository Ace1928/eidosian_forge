from __future__ import annotations
import typing as t
import sqlglot
from sqlglot import expressions as exp
from sqlglot.dataframe.sql.types import DataType
from sqlglot.helper import flatten, is_iterable
@classmethod
def ensure_literal(cls, value) -> Column:
    from sqlglot.dataframe.sql.functions import lit
    if isinstance(value, cls):
        value = value.expression
    if not isinstance(value, exp.Literal):
        return lit(value)
    return Column(value)