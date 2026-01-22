from datetime import (
import itertools
import re
import numpy as np
import pytest
from pandas._libs.internals import BlockPlacement
from pandas.compat import IS64
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.internals import (
from pandas.core.internals.blocks import (
@pytest.mark.parametrize('arr, slc', [([0], slice(0, 1, 1)), ([100], slice(100, 101, 1)), ([0, 1, 2], slice(0, 3, 1)), ([0, 5, 10], slice(0, 15, 5)), ([0, 100], slice(0, 200, 100)), ([2, 1], slice(2, 0, -1))])
def test_array_to_slice_conversion(self, arr, slc):
    assert BlockPlacement(arr).as_slice == slc