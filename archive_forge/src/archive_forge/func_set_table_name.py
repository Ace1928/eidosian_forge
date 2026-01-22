from __future__ import annotations
import typing as t
import sqlglot
from sqlglot import expressions as exp
from sqlglot.dataframe.sql.types import DataType
from sqlglot.helper import flatten, is_iterable
def set_table_name(self, table_name: str, copy=False) -> Column:
    expression = self.expression.copy() if copy else self.expression
    expression.set('table', exp.to_identifier(table_name))
    return Column(expression)