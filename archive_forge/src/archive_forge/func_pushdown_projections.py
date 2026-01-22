from collections import defaultdict
from sqlglot import alias, exp
from sqlglot.optimizer.qualify_columns import Resolver
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import ensure_schema
def pushdown_projections(expression, schema=None, remove_unused_selections=True):
    """
    Rewrite sqlglot AST to remove unused columns projections.

    Example:
        >>> import sqlglot
        >>> sql = "SELECT y.a AS a FROM (SELECT x.a AS a, x.b AS b FROM x) AS y"
        >>> expression = sqlglot.parse_one(sql)
        >>> pushdown_projections(expression).sql()
        'SELECT y.a AS a FROM (SELECT x.a AS a FROM x) AS y'

    Args:
        expression (sqlglot.Expression): expression to optimize
        remove_unused_selections (bool): remove selects that are unused
    Returns:
        sqlglot.Expression: optimized expression
    """
    schema = ensure_schema(schema)
    source_column_alias_count = {}
    referenced_columns = defaultdict(set)
    for scope in reversed(traverse_scope(expression)):
        parent_selections = referenced_columns.get(scope, {SELECT_ALL})
        alias_count = source_column_alias_count.get(scope, 0)
        if scope.expression.args.get('distinct'):
            parent_selections = {SELECT_ALL}
        if isinstance(scope.expression, exp.Union):
            left, right = scope.union_scopes
            referenced_columns[left] = parent_selections
            if any((select.is_star for select in right.expression.selects)):
                referenced_columns[right] = parent_selections
            elif not any((select.is_star for select in left.expression.selects)):
                if scope.expression.args.get('by_name'):
                    referenced_columns[right] = referenced_columns[left]
                else:
                    referenced_columns[right] = [right.expression.selects[i].alias_or_name for i, select in enumerate(left.expression.selects) if SELECT_ALL in parent_selections or select.alias_or_name in parent_selections]
        if isinstance(scope.expression, exp.Select):
            if remove_unused_selections:
                _remove_unused_selections(scope, parent_selections, schema, alias_count)
            if scope.expression.is_star:
                continue
            selects = defaultdict(set)
            for col in scope.columns:
                table_name = col.table
                col_name = col.name
                selects[table_name].add(col_name)
            for name, (node, source) in scope.selected_sources.items():
                if isinstance(source, Scope):
                    columns = {SELECT_ALL} if scope.pivots else selects.get(name) or set()
                    referenced_columns[source].update(columns)
                column_aliases = node.alias_column_names
                if column_aliases:
                    source_column_alias_count[source] = len(column_aliases)
    return expression