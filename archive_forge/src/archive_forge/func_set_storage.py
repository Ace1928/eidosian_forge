import copy
import json
import logging
from contextlib import contextmanager
from functools import partial
from numbers import Number
import os
from pathlib import Path
import platform
import re
import time
from typing import Any, Dict, Optional, Sequence, Union, Callable, List, Tuple
import uuid
import ray
from ray.air.constants import (
import ray.cloudpickle as cloudpickle
from ray.exceptions import RayActorError, RayTaskError
from ray.train import Checkpoint, CheckpointConfig
from ray.train.constants import (
from ray.train._internal.checkpoint_manager import _CheckpointManager
from ray.train._internal.session import _FutureTrainingResult, _TrainingResult
from ray.train._internal.storage import StorageContext
from ray.tune import TuneError
from ray.tune.logger import NoopLogger
from ray.tune.registry import get_trainable_cls, validate_trainable
from ray.tune.result import (
from ray.tune.execution.placement_groups import (
from ray.tune.trainable.metadata import _TrainingRunMetadata
from ray.tune.utils.serialization import TuneFunctionDecoder, TuneFunctionEncoder
from ray.tune.utils import date_str, flatten_dict
from ray.util.annotations import DeveloperAPI, Deprecated
from ray._private.utils import binary_to_hex, hex_to_binary
def set_storage(self, new_storage: StorageContext):
    """Updates the storage context of the trial.

        If the `storage_path` or `experiment_dir_name` has changed, then this setter
        also updates the paths of all checkpoints tracked by the checkpoint manager.
        This enables restoration from a checkpoint if the user moves the directory.
        """
    original_storage = self.storage
    checkpoint_manager = self.run_metadata.checkpoint_manager
    for checkpoint_result in checkpoint_manager.best_checkpoint_results:
        checkpoint_result.checkpoint = Checkpoint(path=checkpoint_result.checkpoint.path.replace(original_storage.trial_fs_path, new_storage.trial_fs_path, 1), filesystem=new_storage.storage_filesystem)
    latest_checkpoint_result = checkpoint_manager.latest_checkpoint_result
    if latest_checkpoint_result:
        latest_checkpoint_result.checkpoint = Checkpoint(path=latest_checkpoint_result.checkpoint.path.replace(original_storage.trial_fs_path, new_storage.trial_fs_path, 1), filesystem=new_storage.storage_filesystem)
    self.storage = new_storage
    self.invalidate_json_state()