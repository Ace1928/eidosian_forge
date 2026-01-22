import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar
import ray
import ray._private.ray_constants as ray_constants
from ray._private.ray_constants import env_integer
from ray.data import Dataset
from ray.exceptions import RayActorError
from ray.train import Checkpoint, DataConfig
from ray.train._internal.session import (
from ray.train._internal.storage import StorageContext
from ray.train._internal.utils import check_for_failure
from ray.train._internal.worker_group import WorkerGroup
from ray.train.backend import BackendConfig
from ray.train.constants import (
from ray.util.placement_group import get_current_placement_group, remove_placement_group
def initialize_session(train_func, world_rank, local_rank, node_rank, local_world_size, world_size, trial_info, checkpoint, dataset_shard, metadata, storage):
    try:
        init_session(training_func=train_func, world_rank=world_rank, local_rank=local_rank, node_rank=node_rank, local_world_size=local_world_size, world_size=world_size, trial_info=trial_info, dataset_shard=dataset_shard, metadata=metadata, checkpoint=checkpoint, detailed_autofilled_metrics=use_detailed_autofilled_metrics, storage=storage)
    except ValueError:
        raise TrainBackendError('Attempting to start training but a previous training run is still ongoing. You must call `finish_training` before calling `start_training` again.')