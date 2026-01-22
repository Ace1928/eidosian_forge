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
@pytest.mark.parametrize('mgr_string', ['a:i8;b:f8', 'a:i8;b:f8;c:c8;d:b', 'a:i8;e:dt;f:td;g:string', 'a:i8;b:category;c:category2', 'c:sparse;d:sparse_na;b:f8'])
def test_equals_block_order_different_dtypes(self, mgr_string):
    bm = create_mgr(mgr_string)
    block_perms = itertools.permutations(bm.blocks)
    for bm_perm in block_perms:
        bm_this = BlockManager(tuple(bm_perm), bm.axes)
        assert bm.equals(bm_this)
        assert bm_this.equals(bm)