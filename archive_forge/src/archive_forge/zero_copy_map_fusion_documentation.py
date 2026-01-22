from abc import abstractmethod
from typing import List
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
from ray.data._internal.logical.interfaces.optimizer import Rule
from ray.data._internal.logical.interfaces.physical_plan import PhysicalPlan
Optimize the transform_fns chain of a MapOperator.

        Args:
            transform_fns: The old transform_fns chain.
        Returns:
            The optimized transform_fns chain.
        