from __future__ import annotations
import math
import typing as t
from sqlglot import alias, exp
from sqlglot.helper import name_sequence
from sqlglot.optimizer.eliminate_joins import join_condition
@classmethod
def from_joins(cls, joins: t.Iterable[exp.Join], ctes: t.Optional[t.Dict[str, Step]]=None) -> Join:
    step = Join()
    for join in joins:
        source_key, join_key, condition = join_condition(join)
        step.joins[join.alias_or_name] = {'side': join.side, 'join_key': join_key, 'source_key': source_key, 'condition': condition}
        step.add_dependency(Scan.from_expression(join.this, ctes))
    return step