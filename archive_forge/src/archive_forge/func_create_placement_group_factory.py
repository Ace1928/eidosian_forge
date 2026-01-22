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
def create_placement_group_factory(self):
    """Compute placement group factory if needed.

        Note: this must be called after all the placeholders in
        self.config are resolved.
        """
    trainable_cls = self.get_trainable_cls()
    if not trainable_cls or not self._setup_default_resource:
        self.placement_group_factory = self._default_placement_group_factory or resource_dict_to_pg_factory()
        return
    default_resources = trainable_cls.default_resource_request(self.config)
    if default_resources and self._default_placement_group_factory:
        raise TuneError('Resources for {} have been automatically set to {} by its `default_resource_request()` method. Please clear the `resources_per_trial` option.'.format(trainable_cls, default_resources))
    if default_resources and (not isinstance(default_resources, PlacementGroupFactory)):
        default_resources = resource_dict_to_pg_factory(default_resources)
    self.placement_group_factory = default_resources or self._default_placement_group_factory or resource_dict_to_pg_factory()