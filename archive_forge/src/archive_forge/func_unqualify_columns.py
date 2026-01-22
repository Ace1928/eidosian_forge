from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name, name_sequence
def unqualify_columns(expression: exp.Expression) -> exp.Expression:
    for column in expression.find_all(exp.Column):
        for part in column.parts[:-1]:
            part.pop()
    return expression