import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
def test_bins_not_overlapping_from_interval_index():
    msg = 'Overlapping IntervalIndex is not accepted'
    ii = IntervalIndex.from_tuples([(0, 10), (2, 12), (4, 14)])
    with pytest.raises(ValueError, match=msg):
        cut([5, 6], bins=ii)