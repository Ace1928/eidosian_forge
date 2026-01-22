import logging
from typing import Dict, Optional, Union, TYPE_CHECKING
import numpy as np
import pickle
from ray.tune.result import DEFAULT_METRIC
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.experiment import Trial
from ray.util import PublicAPI
def on_result(self, trial: Trial, cur_iter: int, cur_rew: Optional[float]) -> str:
    action = TrialScheduler.CONTINUE
    for milestone, recorded in self._rungs:
        if cur_iter >= milestone and trial.trial_id in recorded and (not self._stop_last_trials):
            break
        if cur_iter < milestone or trial.trial_id in recorded:
            continue
        else:
            cutoff = self.cutoff(recorded)
            if cutoff is not None and cur_rew < cutoff:
                action = TrialScheduler.STOP
            if cur_rew is None:
                logger.warning('Reward attribute is None! Consider reporting using a different field.')
            else:
                recorded[trial.trial_id] = cur_rew
            break
    return action