from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.dataframe.sql.column import Column
from sqlglot.dataframe.sql.util import get_tables_from_expression_with_join
from sqlglot.helper import ensure_list
def replace_alias_name_with_cte_name(spark: SparkSession, expression_context: exp.Select, id: exp.Identifier):
    if id.alias_or_name in spark.name_to_sequence_id_mapping:
        for cte in reversed(expression_context.ctes):
            if cte.args['sequence_id'] in spark.name_to_sequence_id_mapping[id.alias_or_name]:
                _set_alias_name(id, cte.alias_or_name)
                break