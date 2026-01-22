import logging
from typing import Optional
import wandb
from wandb.util import (
from . import wandb_helper
from .lib import config_util
def setdefaults(self, d):
    d = wandb_helper.parse_config(d)
    d = {k: v for k, v in d.items() if k not in self._items}
    d = self._sanitize_dict(d)
    self._items.update(d)
    if self._callback:
        self._callback(data=d)