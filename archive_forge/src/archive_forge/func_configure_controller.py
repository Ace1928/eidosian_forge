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
def configure_controller(self, type):
    """Configure controller to local if type == 'local'."""
    self._configure_check()
    self._create.setdefault('controller', {})
    self._create['controller'].setdefault('type', type)