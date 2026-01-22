from __future__ import annotations
import os
from typing import Callable
import pandas._config.config as cf
from pandas._config.config import (
def register_converter_cb(key) -> None:
    from pandas.plotting import deregister_matplotlib_converters, register_matplotlib_converters
    if cf.get_option(key):
        register_matplotlib_converters()
    else:
        deregister_matplotlib_converters()