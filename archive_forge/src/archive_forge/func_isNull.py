from __future__ import annotations
import typing as t
import sqlglot
from sqlglot import expressions as exp
from sqlglot.dataframe.sql.types import DataType
from sqlglot.helper import flatten, is_iterable
def isNull(self) -> Column:
    new_expression = exp.Is(this=self.column_expression, expression=exp.Null())
    return Column(new_expression)