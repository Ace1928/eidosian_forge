from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def qualify_derived_table_outputs(expression: exp.Expression) -> exp.Expression:
    """Ensures all (unnamed) output columns are aliased for CTEs and Subqueries."""
    alias = expression.args.get('alias')
    if isinstance(expression, (exp.CTE, exp.Subquery)) and isinstance(alias, exp.TableAlias) and (not alias.columns):
        from sqlglot.optimizer.qualify_columns import qualify_outputs
        query = expression.this
        unaliased_column_indexes = (i for i, c in enumerate(query.selects) if isinstance(c, exp.Column) and (not c.alias))
        qualify_outputs(query)
        query_selects = query.selects
        for select_index in unaliased_column_indexes:
            alias = query_selects[select_index]
            column = alias.this
            if isinstance(column.this, exp.Identifier):
                alias.args['alias'].set('quoted', column.this.quoted)
    return expression