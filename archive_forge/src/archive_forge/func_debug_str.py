import logging
from typing import Dict, Optional, Union, TYPE_CHECKING
import numpy as np
import pickle
from ray.tune.result import DEFAULT_METRIC
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.experiment import Trial
from ray.util import PublicAPI
def debug_str(self) -> str:
    iters = ' | '.join(['Iter {:.3f}: {}'.format(milestone, self.cutoff(recorded)) for milestone, recorded in self._rungs])
    return 'Bracket: ' + iters