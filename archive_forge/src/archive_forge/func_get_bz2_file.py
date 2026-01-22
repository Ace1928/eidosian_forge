from __future__ import annotations
import os
import platform
import sys
from typing import TYPE_CHECKING
from pandas.compat._constants import (
import pandas.compat.compressors
from pandas.compat.numpy import is_numpy_dev
from pandas.compat.pyarrow import (
def get_bz2_file() -> type[pandas.compat.compressors.BZ2File]:
    """
    Importing the `BZ2File` class from the `bz2` module.

    Returns
    -------
    class
        The `BZ2File` class from the `bz2` module.

    Raises
    ------
    RuntimeError
        If the `bz2` module was not imported correctly, or didn't exist.
    """
    if not pandas.compat.compressors.has_bz2:
        raise RuntimeError('bz2 module not available. A Python re-install with the proper dependencies, might be required to solve this issue.')
    return pandas.compat.compressors.BZ2File