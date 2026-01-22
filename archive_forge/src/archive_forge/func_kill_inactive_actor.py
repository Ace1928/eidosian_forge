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
def kill_inactive_actor(self) -> bool:
    """Kills a single pending or idle actor, if any actors are pending/idle.

        Returns whether an inactive actor was actually killed.
        """
    killed = self._maybe_kill_pending_actor()
    if not killed:
        killed = self._maybe_kill_idle_actor()
    return killed