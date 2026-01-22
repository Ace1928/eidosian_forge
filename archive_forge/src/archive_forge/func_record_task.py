import collections
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4
import numpy as np
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces.op_runtime_metrics import OpRuntimeMetrics
from ray.data._internal.util import capfirst
from ray.data.block import BlockMetadata
from ray.data.context import DataContext
from ray.util.annotations import DeveloperAPI
from ray.util.metrics import Gauge
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def record_task(self, stats_uuid: str, task_idx: int, blocks_metadata: List[BlockMetadata]):
    for metadata in blocks_metadata:
        metadata.schema = None
    if stats_uuid in self.start_time:
        self.metadata[stats_uuid][task_idx] = blocks_metadata
        self.last_time[stats_uuid] = time.perf_counter()