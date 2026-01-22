from typing import Dict
from collections import defaultdict
from ray.util.annotations import PublicAPI
from ray.tune.stopper.stopper import Stopper
Stop trials after reaching a maximum number of iterations

    Args:
        max_iter: Number of iterations before stopping a trial.
    