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
def suggest(self, trial_id: str) -> Optional[Dict]:
    if not self._space:
        raise RuntimeError(UNDEFINED_SEARCH_SPACE.format(cls=self.__class__.__name__, space='space'))
    if not self._metric or not self._mode:
        raise RuntimeError(UNDEFINED_METRIC_MODE.format(cls=self.__class__.__name__, metric=self._metric, mode=self._mode))
    if callable(self._space):
        if trial_id not in self._ot_trials:
            self._ot_trials[trial_id] = self._ot_study.ask()
        ot_trial = self._ot_trials[trial_id]
        params = self._suggest_from_define_by_run_func(self._space, ot_trial)
    else:
        if trial_id not in self._ot_trials:
            self._ot_trials[trial_id] = self._ot_study.ask(fixed_distributions=self._space)
        ot_trial = self._ot_trials[trial_id]
        params = ot_trial.params
    return unflatten_dict(params)