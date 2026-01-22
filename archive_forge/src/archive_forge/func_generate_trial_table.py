from __future__ import print_function
import collections
import datetime
import numbers
import os
import sys
import textwrap
import time
import warnings
from typing import Any, Callable, Collection, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import ray
from ray._private.dict import flatten_dict
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.experimental.tqdm_ray import safe_print
from ray.air.util.node import _force_on_current_node
from ray.air.constants import EXPR_ERROR_FILE, TRAINING_ITERATION
from ray.tune.callback import Callback
from ray.tune.logger import pretty_print
from ray.tune.result import (
from ray.tune.experiment.trial import DEBUG_PRINT_INTERVAL, Trial, _Location
from ray.tune.trainable import Trainable
from ray.tune.utils import unflattened_lookup
from ray.tune.utils.log import Verbosity, has_verbosity, set_verbosity
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.queue import Empty, Queue
from ray.widgets import Template
def generate_trial_table(self, trials: Dict[Trial, Dict], columns: List[str]) -> str:
    """Generate an HTML table of trial progress info.

        Trials (rows) are sorted by name; progress stats (columns) are sorted
        as well.

        Args:
            trials: Trials and their associated latest results
            columns: Columns to show in the table; must be a list of valid
                keys for each Trial result

        Returns:
            HTML template containing a rendered table of progress info
        """
    data = []
    columns = sorted(columns)
    sorted_trials = collections.OrderedDict(sorted(self._last_result.items(), key=lambda item: str(item[0])))
    for trial, result in sorted_trials.items():
        data.append([str(trial)] + [result.get(col, '') for col in columns])
    return Template('trial_progress.html.j2').render(table=tabulate(data, tablefmt='html', headers=['Trial name'] + columns, showindex=False))