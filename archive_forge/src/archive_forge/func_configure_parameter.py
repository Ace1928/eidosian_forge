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
def configure_parameter(self, name, values=None, value=None, distribution=None, min=None, max=None, mu=None, sigma=None, q=None, a=None, b=None):
    self._configure_check()
    self._create.setdefault('parameters', {}).setdefault(name, {})
    if value is not None or (values is None and min is None and (max is None) and (distribution is None)):
        self._create['parameters'][name]['value'] = value
    if values is not None:
        self._create['parameters'][name]['values'] = values
    if min is not None:
        self._create['parameters'][name]['min'] = min
    if max is not None:
        self._create['parameters'][name]['max'] = max
    if mu is not None:
        self._create['parameters'][name]['mu'] = mu
    if sigma is not None:
        self._create['parameters'][name]['sigma'] = sigma
    if q is not None:
        self._create['parameters'][name]['q'] = q
    if a is not None:
        self._create['parameters'][name]['a'] = a
    if b is not None:
        self._create['parameters'][name]['b'] = b