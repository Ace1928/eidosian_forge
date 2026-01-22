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
def test_sparse_mixed(self):
    mgr = create_mgr('a: sparse-1; b: sparse-2; c: f8')
    assert len(mgr.blocks) == 3
    assert isinstance(mgr, BlockManager)