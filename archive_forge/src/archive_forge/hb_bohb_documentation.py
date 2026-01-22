import logging
from typing import Dict, Optional, TYPE_CHECKING
from ray.tune.schedulers.trial_scheduler import TrialScheduler
from ray.tune.schedulers.hyperband import HyperBandScheduler
from ray.tune.experiment import Trial
from ray.util import PublicAPI
Fair scheduling within iteration by completion percentage.

        List of trials not used since all trials are tracked as state
        of scheduler. If iteration is occupied (ie, no trials to run),
        then look into next iteration.
        