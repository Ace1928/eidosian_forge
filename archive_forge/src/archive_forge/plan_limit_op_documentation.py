from typing import TYPE_CHECKING
from ray.data._internal.execution.operators.limit_operator import LimitOperator
Get the corresponding DAG of physical operators for Limit.

    Note this method only converts the given `op`, but not its input dependencies.
    See Planner.plan() for more details.
    