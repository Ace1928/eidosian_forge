import pytest
from pandas import (
def test_contains_infinite_length(self):
    interval1 = Interval(0, 1, 'both')
    interval2 = Interval(float('-inf'), float('inf'), 'neither')
    assert interval1 in interval2
    assert interval2 not in interval1