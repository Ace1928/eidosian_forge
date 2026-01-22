import time
import logging
import pickle
import functools
import warnings
from packaging import version
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ray.air.constants import TRAINING_ITERATION
from ray.tune.result import DEFAULT_METRIC
from ray.tune.search.sample import (
from ray.tune.search import (
from ray.tune.search.variant_generator import parse_spec_vars
from ray.tune.utils.util import flatten_dict, unflatten_dict, validate_warmstart
def set_search_properties(self, metric: Optional[str], mode: Optional[str], config: Dict, **spec) -> bool:
    if self._space:
        return False
    space = self.convert_search_space(config)
    self._space = space
    if metric:
        self._metric = metric
    if mode:
        self._mode = mode
    self._setup_study(self._mode)
    return True