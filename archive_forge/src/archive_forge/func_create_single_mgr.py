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
def create_single_mgr(typestr, num_rows=None):
    if num_rows is None:
        num_rows = N
    return SingleBlockManager(create_block(typestr, placement=slice(0, num_rows), item_shape=()), Index(np.arange(num_rows)))