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
def test_single_mgr_ctor(self):
    mgr = create_single_mgr('f8', num_rows=5)
    assert mgr.external_values().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]