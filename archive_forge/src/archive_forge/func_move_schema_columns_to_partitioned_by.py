from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name, name_sequence
def move_schema_columns_to_partitioned_by(expression: exp.Expression) -> exp.Expression:
    """
    In Hive, the PARTITIONED BY property acts as an extension of a table's schema. When the
    PARTITIONED BY value is an array of column names, they are transformed into a schema.
    The corresponding columns are removed from the create statement.
    """
    assert isinstance(expression, exp.Create)
    has_schema = isinstance(expression.this, exp.Schema)
    is_partitionable = expression.kind in {'TABLE', 'VIEW'}
    if has_schema and is_partitionable:
        prop = expression.find(exp.PartitionedByProperty)
        if prop and prop.this and (not isinstance(prop.this, exp.Schema)):
            schema = expression.this
            columns = {v.name.upper() for v in prop.this.expressions}
            partitions = [col for col in schema.expressions if col.name.upper() in columns]
            schema.set('expressions', [e for e in schema.expressions if e not in partitions])
            prop.replace(exp.PartitionedByProperty(this=exp.Schema(expressions=partitions)))
            expression.set('this', schema)
    return expression