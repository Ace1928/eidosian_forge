import copy
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_explicit_t(self):
    inp = '\n        - 2011-10-02t16:45:00\n        '
    exp = '\n        - 2011-10-02T16:45:00\n        '
    round_trip(inp, exp)