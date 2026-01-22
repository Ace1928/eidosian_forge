from typing import Any, List
import ray
from ray import cloudpickle
@property
def n(self) -> int:
    return self._weight