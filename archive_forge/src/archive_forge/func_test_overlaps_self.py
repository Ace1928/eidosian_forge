import pytest
from pandas import (
def test_overlaps_self(self, start_shift, closed):
    start, shift = start_shift
    interval = Interval(start, start + shift, closed)
    assert interval.overlaps(interval)