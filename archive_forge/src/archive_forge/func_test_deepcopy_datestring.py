import copy
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_deepcopy_datestring(self):
    x = dedent('        foo: 2016-10-12T12:34:56\n        ')
    data = copy.deepcopy(round_trip_load(x))
    assert round_trip_dump(data) == x