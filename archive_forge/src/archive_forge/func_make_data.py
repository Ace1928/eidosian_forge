from __future__ import annotations
from collections import (
import itertools
import numbers
import string
import sys
from typing import (
import numpy as np
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import (
import pandas as pd
from pandas.api.extensions import (
from pandas.core.indexers import unpack_tuple_and_ellipses
def make_data():
    rng = np.random.default_rng(2)
    return [UserDict([(rng.choice(list(string.ascii_letters)), rng.integers(0, 100)) for _ in range(rng.integers(0, 10))]) for _ in range(100)]