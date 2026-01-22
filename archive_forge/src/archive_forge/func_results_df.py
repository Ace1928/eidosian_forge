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
@property
def results_df(self) -> DataFrame:
    """Get all the last results as a pandas dataframe."""
    if not pd:
        raise ValueError('`results_df` requires pandas. Install with `pip install pandas`.')
    return pd.DataFrame.from_records([flatten_dict(trial.last_result, delimiter=self._delimiter()) for trial in self.trials], index='trial_id')