from __future__ import annotations
import math
import typing as t
from sqlglot import alias, exp
from sqlglot.helper import name_sequence
from sqlglot.optimizer.eliminate_joins import join_condition
def extract_agg_operands(expression):
    agg_funcs = tuple(expression.find_all(exp.AggFunc))
    if agg_funcs:
        aggregations.add(expression)
    for agg in agg_funcs:
        for operand in agg.unnest_operands():
            if isinstance(operand, exp.Column):
                continue
            if operand not in operands:
                operands[operand] = next_operand_name()
            operand.replace(exp.column(operands[operand], quoted=True))
    return bool(agg_funcs)