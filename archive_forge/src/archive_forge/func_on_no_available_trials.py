import logging
from functools import lru_cache
import os
import ray
import time
from typing import Dict, Optional, Tuple
from ray.tune.execution.cluster_info import _is_ray_cluster
from ray.tune.experiment import Trial
def on_no_available_trials(self, all_trials):
    """Tracks information across the life of Tune loop and makes guesses
        about if Tune loop is stuck due to infeasible resources.
        If so, outputs certain warning messages.
        The logic should be conservative, non-intrusive and informative.
        For example, rate limiting is applied so that the message is not
        spammy.
        """
    if len(all_trials) == self._last_trial_num:
        if self._no_running_trials_since == -1:
            self._no_running_trials_since = time.monotonic()
        elif time.monotonic() - self._no_running_trials_since > _get_insufficient_resources_warning_threshold():
            can_fulfill_any = any((trial.status == Trial.PENDING and _can_fulfill_no_autoscaler(trial) for trial in all_trials))
            if can_fulfill_any:
                self._no_running_trials_since = -1
                return
            msg = _get_insufficient_resources_warning_msg(for_train=self._for_train, trial=all_trials[0])
            logger.warning(msg)
            self._no_running_trials_since = time.monotonic()
    else:
        self._no_running_trials_since = -1
    self._last_trial_num = len(all_trials)