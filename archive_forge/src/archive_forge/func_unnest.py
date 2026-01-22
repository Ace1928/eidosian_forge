from sqlglot import exp
from sqlglot.helper import name_sequence
from sqlglot.optimizer.scope import ScopeType, traverse_scope
def unnest(select, parent_select, next_alias_name):
    if len(select.selects) > 1:
        return
    predicate = select.find_ancestor(exp.Condition)
    if not predicate or parent_select is not predicate.parent_select or (not parent_select.args.get('from')):
        return
    if isinstance(select, exp.Union):
        select = exp.select(*select.selects).from_(select.subquery(next_alias_name()))
    alias = next_alias_name()
    clause = predicate.find_ancestor(exp.Having, exp.Where, exp.Join)
    if not isinstance(predicate, (exp.In, exp.Any)):
        column = exp.column(select.selects[0].alias_or_name, alias)
        clause_parent_select = clause.parent_select if clause else None
        if isinstance(clause, exp.Having) and clause_parent_select is parent_select or ((not clause or clause_parent_select is not parent_select) and (parent_select.args.get('group') or any((projection.find(exp.AggFunc) for projection in parent_select.selects)))):
            column = exp.Max(this=column)
        elif not isinstance(select.parent, exp.Subquery):
            return
        _replace(select.parent, column)
        parent_select.join(select, join_type='CROSS', join_alias=alias, copy=False)
        return
    if select.find(exp.Limit, exp.Offset):
        return
    if isinstance(predicate, exp.Any):
        predicate = predicate.find_ancestor(exp.EQ)
        if not predicate or parent_select is not predicate.parent_select:
            return
    column = _other_operand(predicate)
    value = select.selects[0]
    join_key = exp.column(value.alias, alias)
    join_key_not_null = join_key.is_(exp.null()).not_()
    if isinstance(clause, exp.Join):
        _replace(predicate, exp.true())
        parent_select.where(join_key_not_null, copy=False)
    else:
        _replace(predicate, join_key_not_null)
    group = select.args.get('group')
    if group:
        if {value.this} != set(group.expressions):
            select = exp.select(exp.column(value.alias, '_q')).from_(select.subquery('_q', copy=False), copy=False).group_by(exp.column(value.alias, '_q'), copy=False)
    else:
        select = select.group_by(value.this, copy=False)
    parent_select.join(select, on=column.eq(join_key), join_type='LEFT', join_alias=alias, copy=False)