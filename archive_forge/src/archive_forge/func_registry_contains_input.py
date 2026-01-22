import atexit
import logging
from functools import partial
from types import FunctionType
from typing import Callable, Optional, Type, Union
import ray
import ray.cloudpickle as pickle
from ray.experimental.internal_kv import (
from ray.tune.error import TuneError
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
def registry_contains_input(name: str) -> bool:
    return _global_registry.contains(RLLIB_INPUT, name)