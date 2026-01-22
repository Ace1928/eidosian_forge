from __future__ import annotations
import os
from typing import Callable
import pandas._config.config as cf
from pandas._config.config import (
def use_bottleneck_cb(key) -> None:
    from pandas.core import nanops
    nanops.set_use_bottleneck(cf.get_option(key))