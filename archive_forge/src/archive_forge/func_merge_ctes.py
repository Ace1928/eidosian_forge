from collections import defaultdict
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name
from sqlglot.optimizer.scope import Scope, traverse_scope
def merge_ctes(expression, leave_tables_isolated=False):
    scopes = traverse_scope(expression)
    cte_selections = defaultdict(list)
    for outer_scope in scopes:
        for table, inner_scope in outer_scope.selected_sources.values():
            if isinstance(inner_scope, Scope) and inner_scope.is_cte:
                cte_selections[id(inner_scope)].append((outer_scope, inner_scope, table))
    singular_cte_selections = [v[0] for k, v in cte_selections.items() if len(v) == 1]
    for outer_scope, inner_scope, table in singular_cte_selections:
        from_or_join = table.find_ancestor(exp.From, exp.Join)
        if _mergeable(outer_scope, inner_scope, leave_tables_isolated, from_or_join):
            alias = table.alias_or_name
            _rename_inner_sources(outer_scope, inner_scope, alias)
            _merge_from(outer_scope, inner_scope, table, alias)
            _merge_expressions(outer_scope, inner_scope, alias)
            _merge_joins(outer_scope, inner_scope, from_or_join)
            _merge_where(outer_scope, inner_scope, from_or_join)
            _merge_order(outer_scope, inner_scope)
            _merge_hints(outer_scope, inner_scope)
            _pop_cte(inner_scope)
            outer_scope.clear_cache()
    return expression