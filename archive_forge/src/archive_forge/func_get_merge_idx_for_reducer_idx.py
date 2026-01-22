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
def get_merge_idx_for_reducer_idx(self, reducer_idx: int) -> int:
    if reducer_idx < (self.merge_partition_size + 1) * self._partitions_with_extra_task:
        merge_idx = reducer_idx // (self.merge_partition_size + 1)
    else:
        reducer_idx -= (self.merge_partition_size + 1) * self._partitions_with_extra_task
        merge_idx = self._partitions_with_extra_task + reducer_idx // self.merge_partition_size
    assert merge_idx < self.num_merge_tasks_per_round
    return merge_idx