import collections
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import logging
from ray.util.annotations import PublicAPI
from ray.tune.result import DEFAULT_METRIC
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.experiment import Trial
from ray.tune.error import TuneError
def filled(self) -> bool:
    """Checks if bracket is filled.

        Only let new trials be added at current level minimizing the need
        to backtrack and bookkeep previous medians."""
    return len(self._live_trials) == self._n