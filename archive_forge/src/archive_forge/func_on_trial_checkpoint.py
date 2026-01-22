from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union
import click
import logging
import os
import time
import warnings
from ray.train._internal.storage import (
from ray.tune.experiment import Trial
from ray.tune.impl.out_of_band_serialize_dataset import out_of_band_serialize_dataset
def on_trial_checkpoint(self, trial: Trial):
    if not self._sync_every_n_trial_checkpoints:
        return
    self._trial_num_checkpoints_since_last_sync[trial] += 1
    if self._trial_num_checkpoints_since_last_sync[trial] >= self._sync_every_n_trial_checkpoints:
        self._should_force_cloud_sync = True