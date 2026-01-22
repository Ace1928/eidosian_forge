import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
@pytest.mark.parametrize('other_constructor', [IntervalArray, IntervalIndex])
def test_overlaps_interval_container(self, constructor, other_constructor):
    interval_container = constructor.from_breaks(range(5))
    other_container = other_constructor.from_breaks(range(5))
    with pytest.raises(NotImplementedError, match='^$'):
        interval_container.overlaps(other_container)