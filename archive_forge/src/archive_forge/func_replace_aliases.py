from sqlglot import exp
from sqlglot.optimizer.normalize import normalized
from sqlglot.optimizer.scope import build_scope, find_in_scope
from sqlglot.optimizer.simplify import simplify
def replace_aliases(source, predicate):
    aliases = {}
    for select in source.selects:
        if isinstance(select, exp.Alias):
            aliases[select.alias] = select.this
        else:
            aliases[select.name] = select

    def _replace_alias(column):
        if isinstance(column, exp.Column) and column.name in aliases:
            return aliases[column.name].copy()
        return column
    return predicate.transform(_replace_alias)