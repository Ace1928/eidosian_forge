from __future__ import annotations
import os
from typing import Callable
import pandas._config.config as cf
from pandas._config.config import (
def table_schema_cb(key) -> None:
    from pandas.io.formats.printing import enable_data_resource_formatter
    enable_data_resource_formatter(cf.get_option(key))