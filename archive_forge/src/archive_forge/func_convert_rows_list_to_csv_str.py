from __future__ import annotations
from decimal import Decimal
import operator
import os
from sys import byteorder
from typing import (
import warnings
import numpy as np
from pandas._config.localization import (
from pandas.compat import pa_version_under10p1
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
from pandas._testing._io import (
from pandas._testing._warnings import (
from pandas._testing.asserters import (
from pandas._testing.compat import (
from pandas._testing.contexts import (
from pandas.core.arrays import (
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import extract_array
def convert_rows_list_to_csv_str(rows_list: list[str]) -> str:
    """
    Convert list of CSV rows to single CSV-formatted string for current OS.

    This method is used for creating expected value of to_csv() method.

    Parameters
    ----------
    rows_list : List[str]
        Each element represents the row of csv.

    Returns
    -------
    str
        Expected output of to_csv() in current OS.
    """
    sep = os.linesep
    return sep.join(rows_list) + sep