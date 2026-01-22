import argparse
import sys
from typing import (
import collections
from dataclasses import dataclass
import datetime
from enum import IntEnum
import logging
import math
import numbers
import numpy as np
import os
import pandas as pd
import textwrap
import time
from ray.air._internal.usage import AirEntrypoint
from ray.train import Checkpoint
from ray.tune.search.sample import Domain
from ray.tune.utils.log import Verbosity
import ray
from ray._private.dict import unflattened_lookup, flatten_dict
from ray._private.thirdparty.tabulate.tabulate import (
from ray.air.constants import TRAINING_ITERATION
from ray.tune.callback import Callback
from ray.tune.result import (
from ray.tune.experiment.trial import Trial
def on_trial_start(self, iteration: int, trials: List[Trial], trial: Trial, **info):
    if self.verbosity < self._start_end_verbosity:
        return
    has_config = bool(trial.config)
    self._start_block(f'trial_{trial}_start')
    if has_config:
        print(f'{self._addressing_tmpl.format(trial)} started with configuration:')
        self._print_config(trial)
    else:
        print(f'{self._addressing_tmpl.format(trial)} started without custom configuration.')