from copy import deepcopy
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Callable, TYPE_CHECKING
import pickle
import warnings
from ray.air.execution.resources.request import _sum_bundles
from ray.util.annotations import PublicAPI
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.experiment import Trial
from ray.tune.execution.placement_groups import PlacementGroupFactory
def set_trial_resources(self, trial: Trial, new_resources: Union[Dict, PlacementGroupFactory]) -> bool:
    """Returns True if new_resources were set."""
    if new_resources:
        logger.info(f'Setting trial {trial} resource to {new_resources} with {new_resources._bundles}')
        trial.placement_group_factory = None
        trial.update_resources(new_resources)
        self._reallocated_trial_ids.add(trial.trial_id)
        return True
    return False