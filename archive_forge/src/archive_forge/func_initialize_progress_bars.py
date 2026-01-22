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
def initialize_progress_bars(self, index: int, verbose_progress: bool) -> int:
    """Create progress bars at the given index (line offset in console).

        For AllToAllOperator, zero or more sub progress bar would be created.
        Return the number of progress bars created for this operator.
        """
    is_all_to_all = isinstance(self.op, AllToAllOperator)
    enabled = verbose_progress or is_all_to_all
    self.progress_bar = ProgressBar('- ' + self.op.name, self.op.num_outputs_total(), index, enabled=enabled)
    if enabled:
        num_bars = 1
        if is_all_to_all:
            num_bars += self.op.initialize_sub_progress_bars(index + 1)
    else:
        num_bars = 0
    return num_bars