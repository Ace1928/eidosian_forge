from enum import Enum
import logging
import numpy as np
import random
from typing import Any, Dict, List, Optional, Union
import ray  # noqa F401
import psutil
from ray.rllib.policy.sample_batch import SampleBatch, concat_samples
from ray.rllib.utils.actor_manager import FaultAwareApply
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.metrics.window_stat import WindowStat
from ray.rllib.utils.replay_buffers.base import ReplayBufferInterface
from ray.rllib.utils.typing import SampleBatchType
from ray.util.annotations import DeveloperAPI
from ray.util.debug import log_once
@DeveloperAPI
def warn_replay_capacity(*, item: SampleBatchType, num_items: int) -> None:
    """Warn if the configured replay buffer capacity is too large."""
    if log_once('replay_capacity'):
        item_size = item.size_bytes()
        psutil_mem = psutil.virtual_memory()
        total_gb = psutil_mem.total / 1000000000.0
        mem_size = num_items * item_size / 1000000000.0
        msg = 'Estimated max memory usage for replay buffer is {} GB ({} batches of size {}, {} bytes each), available system memory is {} GB'.format(mem_size, num_items, item.count, item_size, total_gb)
        if mem_size > total_gb:
            raise ValueError(msg)
        elif mem_size > 0.2 * total_gb:
            logger.warning(msg)
        else:
            logger.info(msg)