import itertools
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name
from sqlglot.optimizer.scope import build_scope
def eliminate_subqueries(expression):
    """
    Rewrite derived tables as CTES, deduplicating if possible.

    Example:
        >>> import sqlglot
        >>> expression = sqlglot.parse_one("SELECT a FROM (SELECT * FROM x) AS y")
        >>> eliminate_subqueries(expression).sql()
        'WITH y AS (SELECT * FROM x) SELECT a FROM y AS y'

    This also deduplicates common subqueries:
        >>> expression = sqlglot.parse_one("SELECT a FROM (SELECT * FROM x) AS y CROSS JOIN (SELECT * FROM x) AS z")
        >>> eliminate_subqueries(expression).sql()
        'WITH y AS (SELECT * FROM x) SELECT a FROM y AS y CROSS JOIN y AS z'

    Args:
        expression (sqlglot.Expression): expression
    Returns:
        sqlglot.Expression: expression
    """
    if isinstance(expression, exp.Subquery):
        eliminate_subqueries(expression.this)
        return expression
    root = build_scope(expression)
    if not root:
        return expression
    taken = {}
    for scope in root.cte_scopes:
        taken[scope.expression.parent.alias] = scope
    for scope in root.traverse():
        taken.update({source.name: source for _, source in scope.sources.items() if isinstance(source, exp.Table)})
    existing_ctes = {}
    with_ = root.expression.args.get('with')
    recursive = False
    if with_:
        recursive = with_.args.get('recursive')
        for cte in with_.expressions:
            existing_ctes[cte.this] = cte.alias
    new_ctes = []
    for cte_scope in root.cte_scopes:
        for scope in cte_scope.traverse():
            if scope is cte_scope:
                continue
            new_cte = _eliminate(scope, existing_ctes, taken)
            if new_cte:
                new_ctes.append(new_cte)
        new_ctes.append(cte_scope.expression.parent)
    for scope in itertools.chain(root.union_scopes, root.subquery_scopes, root.table_scopes):
        for child_scope in scope.traverse():
            new_cte = _eliminate(child_scope, existing_ctes, taken)
            if new_cte:
                new_ctes.append(new_cte)
    if new_ctes:
        expression.set('with', exp.With(expressions=new_ctes, recursive=recursive))
    return expression