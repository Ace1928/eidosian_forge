import functools
import logging
import os
import platform
import queue
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set, Type
import ray
from ray.air._internal.session import _get_session
from ray.air._internal.util import RunnerThread, StartTraceback
from ray.air.constants import (
from ray.data import Dataset
from ray.train import Checkpoint
from ray.train._internal.accelerator import Accelerator
from ray.train._internal.storage import StorageContext
from ray.train.constants import (
from ray.train.error import SessionMisuseError
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.debug import log_once
from ray.util.placement_group import _valid_resource_shape
from ray.util.scheduling_strategies import (
def get_dataset_shard(self, dataset_name: Optional[str]=None) -> Optional['DataIterator']:
    shard = self.dataset_shard
    if shard is None:
        warnings.warn('No dataset passed in. Returning None. Make sure to pass in a Dataset to Trainer.run to use this function.')
    elif isinstance(shard, dict):
        if not dataset_name:
            raise RuntimeError('Multiple datasets were passed into ``Trainer``, but no ``dataset_name`` is passed into ``get_dataset_shard``. Please specify which dataset shard to retrieve.')
        return shard.get(dataset_name)
    return shard