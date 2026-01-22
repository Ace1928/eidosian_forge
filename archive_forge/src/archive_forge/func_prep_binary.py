from __future__ import annotations
from collections import defaultdict
from typing import cast
import numpy as np
from pandas.core.dtypes.generic import (
from pandas.core.indexes.api import MultiIndex
def prep_binary(arg1, arg2):
    X = arg1 + 0 * arg2
    Y = arg2 + 0 * arg1
    return (X, Y)