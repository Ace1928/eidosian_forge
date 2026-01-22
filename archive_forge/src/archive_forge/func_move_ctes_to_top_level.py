from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name, name_sequence
def move_ctes_to_top_level(expression: exp.Expression) -> exp.Expression:
    """
    Some dialects (e.g. Hive, T-SQL, Spark prior to version 3) only allow CTEs to be
    defined at the top-level, so for example queries like:

        SELECT * FROM (WITH t(c) AS (SELECT 1) SELECT * FROM t) AS subq

    are invalid in those dialects. This transformation can be used to ensure all CTEs are
    moved to the top level so that the final SQL code is valid from a syntax standpoint.

    TODO: handle name clashes whilst moving CTEs (it can get quite tricky & costly).
    """
    top_level_with = expression.args.get('with')
    for node in expression.find_all(exp.With):
        if node.parent is expression:
            continue
        inner_with = node.pop()
        if not top_level_with:
            top_level_with = inner_with
            expression.set('with', top_level_with)
        else:
            if inner_with.recursive:
                top_level_with.set('recursive', True)
            top_level_with.set('expressions', inner_with.expressions + top_level_with.expressions)
    return expression