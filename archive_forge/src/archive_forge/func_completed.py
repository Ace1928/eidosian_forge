from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
import ray
from .ref_bundle import RefBundle
from ray._raylet import ObjectRefGenerator
from ray.data._internal.execution.interfaces.execution_options import (
from ray.data._internal.execution.interfaces.op_runtime_metrics import OpRuntimeMetrics
from ray.data._internal.logical.interfaces import Operator
from ray.data._internal.stats import StatsDict
from ray.data.context import DataContext
def completed(self) -> bool:
    """Return True when this operator is completed.

        An operator is completed if any of the following conditions are met:
        - All upstream operators are completed and all outputs are taken.
        - All downstream operators are completed.
        """
    return self._inputs_complete and self.num_active_tasks() == 0 and (not self.has_next()) or self._dependents_complete