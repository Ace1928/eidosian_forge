import abc
import json
import logging
import os
import pyarrow
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Set, Type
import yaml
from ray.air._internal.json import SafeFallbackEncoder
from ray.tune.callback import Callback
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
def log_trial_start(self, trial: 'Trial'):
    trial.init_local_path()
    for logger_class in self.logger_classes:
        trial_loggers = self._class_trial_loggers.get(logger_class, {})
        if trial not in trial_loggers:
            logger = logger_class(trial.config, trial.local_path, trial)
            trial_loggers[trial] = logger
        self._class_trial_loggers[logger_class] = trial_loggers