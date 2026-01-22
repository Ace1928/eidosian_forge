from __future__ import annotations
from datetime import datetime
import gc
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BaseMaskedArray
@pytest.mark.arm_slow
def test_engine_reference_cycle(self, simple_index):
    index = simple_index
    nrefs_pre = len(gc.get_referrers(index))
    index._engine
    assert len(gc.get_referrers(index)) == nrefs_pre