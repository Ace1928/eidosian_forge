from __future__ import annotations
import itertools
import typing as t
from sqlglot import alias, exp
from sqlglot.dialects.dialect import Dialect, DialectType
from sqlglot.errors import OptimizeError
from sqlglot.helper import seq_get, SingleValuedMapping
from sqlglot.optimizer.annotate_types import TypeAnnotator
from sqlglot.optimizer.scope import Scope, build_scope, traverse_scope, walk_in_scope
from sqlglot.optimizer.simplify import simplify_parens
from sqlglot.schema import Schema, ensure_schema
def get_source_columns(self, name: str, only_visible: bool=False) -> t.Sequence[str]:
    """Resolve the source columns for a given source `name`."""
    if name not in self.scope.sources:
        raise OptimizeError(f'Unknown table: {name}')
    source = self.scope.sources[name]
    if isinstance(source, exp.Table):
        columns = self.schema.column_names(source, only_visible)
    elif isinstance(source, Scope) and isinstance(source.expression, (exp.Values, exp.Unnest)):
        columns = source.expression.named_selects
        if self.schema.dialect == 'bigquery':
            if source.expression.is_type(exp.DataType.Type.STRUCT):
                for k in source.expression.type.expressions:
                    columns.append(k.name)
    else:
        columns = source.expression.named_selects
    node, _ = self.scope.selected_sources.get(name) or (None, None)
    if isinstance(node, Scope):
        column_aliases = node.expression.alias_column_names
    elif isinstance(node, exp.Expression):
        column_aliases = node.alias_column_names
    else:
        column_aliases = []
    if column_aliases:
        return [alias or name for name, alias in itertools.zip_longest(columns, column_aliases)]
    return columns