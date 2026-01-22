import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_intersection_cases(self, sort):
    base = period_range('6/1/2000', '6/30/2000', freq='D', name='idx')
    rng2 = period_range('5/15/2000', '6/20/2000', freq='D', name='idx')
    expected2 = period_range('6/1/2000', '6/20/2000', freq='D', name='idx')
    rng3 = period_range('5/15/2000', '6/20/2000', freq='D', name='other')
    expected3 = period_range('6/1/2000', '6/20/2000', freq='D', name=None)
    rng4 = period_range('7/1/2000', '7/31/2000', freq='D', name='idx')
    expected4 = PeriodIndex([], name='idx', freq='D')
    for rng, expected in [(rng2, expected2), (rng3, expected3), (rng4, expected4)]:
        result = base.intersection(rng, sort=sort)
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name
        assert result.freq == expected.freq
    base = PeriodIndex(['2011-01-05', '2011-01-04', '2011-01-02', '2011-01-03'], freq='D', name='idx')
    rng2 = PeriodIndex(['2011-01-04', '2011-01-02', '2011-02-02', '2011-02-03'], freq='D', name='idx')
    expected2 = PeriodIndex(['2011-01-04', '2011-01-02'], freq='D', name='idx')
    rng3 = PeriodIndex(['2011-01-04', '2011-01-02', '2011-02-02', '2011-02-03'], freq='D', name='other')
    expected3 = PeriodIndex(['2011-01-04', '2011-01-02'], freq='D', name=None)
    rng4 = period_range('7/1/2000', '7/31/2000', freq='D', name='idx')
    expected4 = PeriodIndex([], freq='D', name='idx')
    for rng, expected in [(rng2, expected2), (rng3, expected3), (rng4, expected4)]:
        result = base.intersection(rng, sort=sort)
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name
        assert result.freq == 'D'
    rng = date_range('6/1/2000', '6/15/2000', freq='min')
    result = rng[0:0].intersection(rng)
    assert len(result) == 0
    result = rng.intersection(rng[0:0])
    assert len(result) == 0