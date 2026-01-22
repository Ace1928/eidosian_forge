import copy
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_spaced_timezone(self):
    inp = '\n        - 2011-10-02T11:45:00 -5\n        '
    exp = '\n        - 2011-10-02T11:45:00-5\n        '
    round_trip(inp, exp)