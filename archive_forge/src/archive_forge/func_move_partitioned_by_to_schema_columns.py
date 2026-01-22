from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name, name_sequence
def move_partitioned_by_to_schema_columns(expression: exp.Expression) -> exp.Expression:
    """
    Spark 3 supports both "HIVEFORMAT" and "DATASOURCE" formats for CREATE TABLE.

    Currently, SQLGlot uses the DATASOURCE format for Spark 3.
    """
    assert isinstance(expression, exp.Create)
    prop = expression.find(exp.PartitionedByProperty)
    if prop and prop.this and isinstance(prop.this, exp.Schema) and all((isinstance(e, exp.ColumnDef) and e.kind for e in prop.this.expressions)):
        prop_this = exp.Tuple(expressions=[exp.to_identifier(e.this) for e in prop.this.expressions])
        schema = expression.this
        for e in prop.this.expressions:
            schema.append('expressions', e)
        prop.set('this', prop_this)
    return expression