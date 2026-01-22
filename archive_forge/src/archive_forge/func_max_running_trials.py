from collections import defaultdict
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict
from ray.tune.callback import Callback
from ray.tune.experiment import Trial
def max_running_trials(self) -> int:
    """Outputs the max number of running trials at a given time.

        Usually used to assert certain number given resource restrictions.
        """
    result = 0
    for snapshot in self._snapshot:
        count = 0
        for trial_id in snapshot:
            if snapshot[trial_id] == Trial.RUNNING:
                count += 1
        result = max(result, count)
    return result