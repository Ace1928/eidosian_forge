import copy
import logging
import math
from ray import cloudpickle
from typing import Dict, List, Optional, Union
from ray.tune.result import DEFAULT_METRIC
from ray.tune.search.sample import (
from ray.tune.search import (
from ray.tune.search.variant_generator import parse_spec_vars
from ray.tune.utils.util import flatten_dict, unflatten_list_dict
def to_wrapper(self, trial_id: str, result: Dict) -> _BOHBJobWrapper:
    return _BOHBJobWrapper(self._metric_op * result[self.metric], result['hyperband_info']['budget'], self.trial_to_params[trial_id])