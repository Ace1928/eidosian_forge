from __future__ import annotations
import os
import platform
import sys
from typing import TYPE_CHECKING
from pandas.compat._constants import (
import pandas.compat.compressors
from pandas.compat.numpy import is_numpy_dev
from pandas.compat.pyarrow import (
def is_ci_environment() -> bool:
    """
    Checking if running in a continuous integration environment by checking
    the PANDAS_CI environment variable.

    Returns
    -------
    bool
        True if the running in a continuous integration environment.
    """
    return os.environ.get('PANDAS_CI', '0') == '1'