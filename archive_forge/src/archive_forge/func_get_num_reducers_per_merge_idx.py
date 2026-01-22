import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import ray
from ray.data._internal.execution.interfaces import RefBundle, TaskContext
from ray.data._internal.planner.exchange.interfaces import (
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def get_num_reducers_per_merge_idx(self, merge_idx: int) -> int:
    """
        Each intermediate merge task will produce outputs for a partition of P
        final reduce tasks. This helper function returns P based on the merge
        task index.
        """
    assert merge_idx < self.num_merge_tasks_per_round
    partition_size = self.merge_partition_size
    if merge_idx < self._partitions_with_extra_task:
        partition_size += 1
    return partition_size