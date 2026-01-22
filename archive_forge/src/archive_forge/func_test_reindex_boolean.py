from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
def test_reindex_boolean(self):
    frame = DataFrame(np.ones((10, 2), dtype=bool), index=np.arange(0, 20, 2), columns=[0, 2])
    reindexed = frame.reindex(np.arange(10))
    assert reindexed.values.dtype == np.object_
    assert isna(reindexed[0][1])
    reindexed = frame.reindex(columns=range(3))
    assert reindexed.values.dtype == np.object_
    assert isna(reindexed[1]).all()