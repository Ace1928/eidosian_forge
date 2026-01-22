import logging
from contextlib import contextmanager
from typing import Dict, Optional, Set
import ray
from ray.tune.error import TuneError
from ray.util.annotations import Deprecated
from ray.util.placement_group import _valid_resource_shape
from ray.util.scheduling_strategies import (
from ray import tune     ->     from ray import train
from ray.train import Checkpoint
@Deprecated
def is_session_enabled() -> bool:
    return False