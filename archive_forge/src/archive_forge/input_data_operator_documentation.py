from typing import Callable, List, Optional
from ray.data._internal.execution.interfaces import RefBundle
from ray.data._internal.logical.interfaces import LogicalOperator
Logical operator for input data.

    This may hold cached blocks from a previous Dataset execution, or
    the arguments for read tasks.
    