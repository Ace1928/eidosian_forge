import collections
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import logging
from ray.util.annotations import PublicAPI
from ray.tune.result import DEFAULT_METRIC
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.experiment import Trial
from ray.tune.error import TuneError
def successive_halving(self, metric: str, metric_op: float) -> Tuple[List[Trial], List[Trial]]:
    if self._halves == 0 and (not self.stop_last_trials):
        return (self._live_trials, [])
    assert self._halves > 0
    self._halves -= 1
    self._n = int(np.ceil(self._n / self._eta))
    self._r *= self._eta
    self._r = int(min(self._r, self._max_t_attr))
    self._cumul_r = self._r
    sorted_trials = sorted(self._live_trials, key=lambda t: metric_op * self._live_trials[t][metric])
    good, bad = (sorted_trials[-self._n:], sorted_trials[:-self._n])
    return (good, bad)