from __future__ import annotations
import os
from typing import Callable
import pandas._config.config as cf
from pandas._config.config import (
def use_numexpr_cb(key) -> None:
    from pandas.core.computation import expressions
    expressions.set_use_numexpr(cf.get_option(key))