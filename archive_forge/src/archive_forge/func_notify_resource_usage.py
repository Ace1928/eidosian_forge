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
def notify_resource_usage(self, input_queue_size: int, under_resource_limits: bool) -> None:
    free_slots = self._actor_pool.num_free_slots()
    if input_queue_size > free_slots and under_resource_limits:
        self._scale_up_if_needed()
    else:
        self._scale_down_if_needed()