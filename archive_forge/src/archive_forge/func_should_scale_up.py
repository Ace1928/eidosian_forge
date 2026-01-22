import collections
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union
import ray
from ray.data._internal.compute import ActorPoolStrategy
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.map_operator import MapOperator, _map_task
from ray.data._internal.execution.operators.map_transformer import MapTransformer
from ray.data._internal.execution.util import locality_string
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
def should_scale_up(self, num_total_workers: int, num_running_workers: int) -> bool:
    """Whether the actor pool should scale up by adding a new actor.

        Args:
            num_total_workers: Total number of workers in actor pool.
            num_running_workers: Number of currently running workers in actor pool.

        Returns:
            Whether the actor pool should be scaled up by one actor.
        """
    if num_total_workers < self._config.min_workers:
        return True
    else:
        return num_total_workers < self._config.max_workers and num_total_workers > 0 and (num_running_workers / num_total_workers > self._config.ready_to_total_workers_ratio)