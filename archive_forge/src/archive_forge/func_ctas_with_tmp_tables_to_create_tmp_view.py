from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name, name_sequence
def ctas_with_tmp_tables_to_create_tmp_view(expression: exp.Expression, tmp_storage_provider: t.Callable[[exp.Expression], exp.Expression]=lambda e: e) -> exp.Expression:
    assert isinstance(expression, exp.Create)
    properties = expression.args.get('properties')
    temporary = any((isinstance(prop, exp.TemporaryProperty) for prop in (properties.expressions if properties else [])))
    if expression.kind == 'TABLE' and temporary:
        if expression.expression:
            return exp.Create(kind='TEMPORARY VIEW', this=expression.this, expression=expression.expression)
        return tmp_storage_provider(expression)
    return expression