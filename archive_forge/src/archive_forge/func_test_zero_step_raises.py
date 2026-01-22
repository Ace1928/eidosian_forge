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
@pytest.mark.parametrize('slc', [slice(1, 1, 0), slice(1, 2, 0)])
def test_zero_step_raises(self, slc):
    msg = 'slice step cannot be zero'
    with pytest.raises(ValueError, match=msg):
        BlockPlacement(slc)