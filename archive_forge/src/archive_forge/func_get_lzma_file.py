from __future__ import annotations
import os
import platform
import sys
from typing import TYPE_CHECKING
from pandas.compat._constants import (
import pandas.compat.compressors
from pandas.compat.numpy import is_numpy_dev
from pandas.compat.pyarrow import (
def get_lzma_file() -> type[pandas.compat.compressors.LZMAFile]:
    """
    Importing the `LZMAFile` class from the `lzma` module.

    Returns
    -------
    class
        The `LZMAFile` class from the `lzma` module.

    Raises
    ------
    RuntimeError
        If the `lzma` module was not imported correctly, or didn't exist.
    """
    if not pandas.compat.compressors.has_lzma:
        raise RuntimeError('lzma module not available. A Python re-install with the proper dependencies, might be required to solve this issue.')
    return pandas.compat.compressors.LZMAFile