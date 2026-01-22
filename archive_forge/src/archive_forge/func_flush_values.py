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
def flush_values(self):
    self._register_atexit()
    for (category, key), value in self._to_flush.items():
        _internal_kv_put(_make_key(self.prefix, category, key), value, overwrite=True)
        self._registered.add((category, key))
    self._to_flush.clear()