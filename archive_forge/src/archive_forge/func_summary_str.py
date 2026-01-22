import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import ray
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.autoscaling_requester import (
from ray.data._internal.execution.backpressure_policy import BackpressurePolicy
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.interfaces.physical_operator import (
from ray.data._internal.execution.operators.base_physical_operator import (
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.util import memory_string
from ray.data._internal.progress_bar import ProgressBar
from ray.data.context import DataContext
def summary_str(self) -> str:
    queued = self.num_queued() + self.op.internal_queue_size()
    active = self.op.num_active_tasks()
    desc = f'- {self.op.name}: {active} active, {queued} queued'
    mem = memory_string((self.op.current_resource_usage().object_store_memory or 0) + self.inqueue_memory_usage())
    desc += f', {mem} objects'
    suffix = self.op.progress_str()
    if suffix:
        desc += f', {suffix}'
    return desc