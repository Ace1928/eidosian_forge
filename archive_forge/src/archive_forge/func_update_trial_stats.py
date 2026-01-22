import collections
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import logging
from ray.util.annotations import PublicAPI
from ray.tune.result import DEFAULT_METRIC
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.experiment import Trial
from ray.tune.error import TuneError
def update_trial_stats(self, trial: Trial, result: Dict):
    """Update result for trial. Called after trial has finished
        an iteration - will decrement iteration count.

        TODO(rliaw): The other alternative is to keep the trials
        in and make sure they're not set as pending later."""
    assert trial in self._live_trials
    assert self._get_result_time(result) >= 0
    observed_time = self._get_result_time(result)
    last_observed = self._get_result_time(self._live_trials[trial])
    delta = observed_time - last_observed
    if delta <= 0:
        logger.info('Restoring from a previous point in time. Previous={}; Now={}'.format(last_observed, observed_time))
    self._completed_progress += delta
    self._live_trials[trial] = result
    self.trials_to_unpause.discard(trial)