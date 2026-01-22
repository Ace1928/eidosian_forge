import copy
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_date_only(self):
    inp = '\n        - 2011-10-02\n        '
    exp = '\n        - 2011-10-02\n        '
    round_trip(inp, exp)