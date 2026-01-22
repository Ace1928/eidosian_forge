from __future__ import annotations
import typing as t
from sqlglot import exp
from sqlglot.helper import tsort
def optimize_joins(expression):
    """
    Removes cross joins if possible and reorder joins based on predicate dependencies.

    Example:
        >>> from sqlglot import parse_one
        >>> optimize_joins(parse_one("SELECT * FROM x CROSS JOIN y JOIN z ON x.a = z.a AND y.a = z.a")).sql()
        'SELECT * FROM x JOIN z ON x.a = z.a AND TRUE JOIN y ON y.a = z.a'
    """
    for select in expression.find_all(exp.Select):
        references = {}
        cross_joins = []
        for join in select.args.get('joins', []):
            tables = other_table_names(join)
            if tables:
                for table in tables:
                    references[table] = references.get(table, []) + [join]
            else:
                cross_joins.append((join.alias_or_name, join))
        for name, join in cross_joins:
            for dep in references.get(name, []):
                on = dep.args['on']
                if isinstance(on, exp.Connector):
                    if len(other_table_names(dep)) < 2:
                        continue
                    operator = type(on)
                    for predicate in on.flatten():
                        if name in exp.column_table_names(predicate):
                            predicate.replace(exp.true())
                            predicate = exp._combine([join.args.get('on'), predicate], operator, copy=False)
                            join.on(predicate, append=False, copy=False)
    expression = reorder_joins(expression)
    expression = normalize(expression)
    return expression