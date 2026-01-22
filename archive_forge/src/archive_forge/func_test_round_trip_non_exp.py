from __future__ import print_function, absolute_import, division, unicode_literals
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_round_trip_non_exp(self):
    data = round_trip('        - 1.0\n        - 1.00\n        - 23.100\n        - -1.0\n        - -1.00\n        - -23.100\n        - 42.\n        - -42.\n        - +42.\n        - .5\n        - +.5\n        - -.5\n        ')
    print(data)
    assert 0.999 < data[0] < 1.001
    assert 0.999 < data[1] < 1.001
    assert 23.099 < data[2] < 23.101
    assert 0.999 < -data[3] < 1.001
    assert 0.999 < -data[4] < 1.001
    assert 23.099 < -data[5] < 23.101
    assert 41.999 < data[6] < 42.001
    assert 41.999 < -data[7] < 42.001
    assert 41.999 < data[8] < 42.001
    assert 0.49 < data[9] < 0.51
    assert 0.49 < data[10] < 0.51
    assert -0.51 < data[11] < -0.49