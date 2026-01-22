import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import Hour
@pytest.mark.parametrize('rng, expected', [(TimedeltaIndex(['5 hour', '2 hour', '4 hour', '9 hour'], name='idx'), TimedeltaIndex(['2 hour', '4 hour'], name='idx')), (TimedeltaIndex(['2 hour', '5 hour', '5 hour', '1 hour'], name='other'), TimedeltaIndex(['1 hour', '2 hour'], name=None)), (TimedeltaIndex(['1 hour', '2 hour', '4 hour', '3 hour'], name='idx')[::-1], TimedeltaIndex(['1 hour', '2 hour', '4 hour', '3 hour'], name='idx'))])
def test_intersection_non_monotonic(self, rng, expected, sort):
    base = TimedeltaIndex(['1 hour', '2 hour', '4 hour', '3 hour'], name='idx')
    result = base.intersection(rng, sort=sort)
    if sort is None:
        expected = expected.sort_values()
    tm.assert_index_equal(result, expected)
    assert result.name == expected.name
    if all(base == rng[::-1]) and sort is None:
        assert isinstance(result.freq, Hour)
    else:
        assert result.freq is None