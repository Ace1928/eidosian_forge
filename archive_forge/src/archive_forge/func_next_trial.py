import copy
import glob
import itertools
import os
import uuid
from typing import Dict, List, Optional, Union, TYPE_CHECKING
import warnings
import numpy as np
from ray.air._internal.usage import tag_searcher
from ray.tune.error import TuneError
from ray.tune.experiment.config_parser import _make_parser, _create_trial_from_spec
from ray.tune.search.sample import np_random_generator, _BackwardsCompatibleNumpyRng
from ray.tune.search.variant_generator import (
from ray.tune.search.search_algorithm import SearchAlgorithm
from ray.tune.utils.util import _atomic_save, _load_newest_checkpoint
from ray.util import PublicAPI
def next_trial(self):
    """Provides one Trial object to be queued into the TrialRunner.

        Returns:
            Trial: Returns a single trial.
        """
    if self.is_finished():
        return None
    if self.max_concurrent > 0 and len(self._live_trials) >= self.max_concurrent:
        return None
    if not self._trial_iter:
        self._trial_iter = iter(self._trial_generator)
    try:
        trial = next(self._trial_iter)
        self._live_trials.add(trial.trial_id)
        return trial
    except StopIteration:
        self._trial_generator = []
        self._trial_iter = None
        self.set_finished()
        return None