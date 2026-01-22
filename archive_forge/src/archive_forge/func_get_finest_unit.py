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
def get_finest_unit(left: str, right: str):
    """
    Find the higher of two datetime64 units.
    """
    if _UNITS.index(left) >= _UNITS.index(right):
        return left
    return right