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
def on_trial_complete(self, trial_id: str, result: Optional[Dict]=None, error: bool=False):
    if trial_id in self._completed_trials:
        logger.warning(f'Received additional completion for trial {trial_id}, but it already finished. Result: {result}')
        return
    ot_trial = self._ot_trials[trial_id]
    if result:
        if isinstance(self.metric, list):
            val = [result.get(metric, None) for metric in self.metric]
        else:
            val = result.get(self.metric, None)
    else:
        val = None
    ot_trial_state = OptunaTrialState.COMPLETE
    if val is None:
        if error:
            ot_trial_state = OptunaTrialState.FAIL
        else:
            ot_trial_state = OptunaTrialState.PRUNED
    try:
        self._ot_study.tell(ot_trial, val, state=ot_trial_state)
    except Exception as exc:
        logger.warning(exc)
    self._completed_trials.add(trial_id)