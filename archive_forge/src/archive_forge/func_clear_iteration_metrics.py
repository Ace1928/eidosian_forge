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
def clear_iteration_metrics(self, dataset_tag: str):
    with self._stats_lock:
        if dataset_tag in self._last_iteration_stats:
            del self._last_iteration_stats[dataset_tag]
    try:
        self._stats_actor(create_if_not_exists=False).clear_iteration_metrics.remote(dataset_tag)
    except Exception:
        pass