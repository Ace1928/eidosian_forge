import copy
import fnmatch
import io
import json
import logging
from numbers import Number
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pyarrow.fs
from ray.util.annotations import PublicAPI
from ray.air.constants import (
from ray.train import Checkpoint
from ray.train._internal.storage import (
from ray.tune.execution.tune_controller import TuneController
from ray.tune.experiment import Trial
from ray.tune.result import (
from ray.tune.utils import flatten_dict
from ray.tune.utils.serialization import TuneFunctionDecoder
from ray.tune.utils.util import is_nan_or_inf, is_nan, unflattened_lookup
def get_trial_checkpoints_paths(self, trial: Trial, metric: Optional[str]=None) -> List[Tuple[str, Number]]:
    raise DeprecationWarning('`get_trial_checkpoints_paths` is deprecated. Use `get_best_checkpoint` or wrap this `ExperimentAnalysis` in a `ResultGrid` and use `Result.best_checkpoints` instead.')