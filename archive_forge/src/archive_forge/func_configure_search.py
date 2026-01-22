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
def configure_search(self, search: Union[str, Callable[[Union[dict, sweeps.SweepConfig], List[sweeps.SweepRun]], Optional[sweeps.SweepRun]]]):
    self._configure_check()
    if isinstance(search, str):
        self._create['method'] = search
    elif callable(search):
        self._create['method'] = 'custom'
        self._custom_search = search
    else:
        raise ControllerError('Unhandled search type.')