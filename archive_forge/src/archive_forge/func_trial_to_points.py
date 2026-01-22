import copy
import glob
import logging
import os
import warnings
from typing import Dict, Optional, List, Union, Any, TYPE_CHECKING
from ray.air._internal.usage import tag_searcher
from ray.tune.search.util import _set_search_properties_backwards_compatible
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.debug import log_once
def trial_to_points(trial: Trial) -> Dict[str, Any]:
    nonlocal any_trial_had_metric
    has_trial_been_pruned = trial.status == Trial.TERMINATED and (not trial.last_result.get(DONE, False))
    has_trial_finished = trial.status == Trial.TERMINATED and trial.last_result.get(DONE, False)
    if not any_trial_had_metric:
        any_trial_had_metric = metric in trial.last_result and has_trial_finished
    if Trial.TERMINATED and metric not in trial.last_result:
        return None
    return dict(parameters=trial.config, value=trial.last_result.get(metric, None), error=trial.status == Trial.ERROR, pruned=has_trial_been_pruned, intermediate_values=None)