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
def test_blockplacement_add(self):
    bpl = BlockPlacement(slice(0, 5))
    assert bpl.add(1).as_slice == slice(1, 6, 1)
    assert bpl.add(np.arange(5)).as_slice == slice(0, 10, 2)
    assert list(bpl.add(np.arange(5, 0, -1))) == [5, 5, 5, 5, 5]