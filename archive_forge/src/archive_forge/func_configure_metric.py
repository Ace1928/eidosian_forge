import json
import os
import random
import string
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import yaml
from wandb import env
from wandb.apis import InternalApi
from wandb.sdk import wandb_sweep
from wandb.sdk.launch.sweeps.utils import (
from wandb.util import get_module
def configure_metric(self, metric, goal=None):
    self._configure_check()
    self._create.setdefault('metric', {})
    self._create['metric']['name'] = metric
    if goal:
        self._create['metric']['goal'] = goal