import pytest
from pandas import Timedelta
def test_sub_day(self):
    delta_1d = Timedelta(1, unit='D')
    delta_0d = Timedelta(0, unit='D')
    delta_1s = Timedelta(1, unit='s')
    delta_500ms = Timedelta(500, unit='ms')
    drepr = lambda x: x._repr_base(format='sub_day')
    assert drepr(delta_1d) == '1 days'
    assert drepr(-delta_1d) == '-1 days'
    assert drepr(delta_0d) == '00:00:00'
    assert drepr(delta_1s) == '00:00:01'
    assert drepr(delta_500ms) == '00:00:00.500000'
    assert drepr(delta_1d + delta_1s) == '1 days 00:00:01'
    assert drepr(-delta_1d + delta_1s) == '-1 days +00:00:01'
    assert drepr(delta_1d + delta_500ms) == '1 days 00:00:00.500000'
    assert drepr(-delta_1d + delta_500ms) == '-1 days +00:00:00.500000'