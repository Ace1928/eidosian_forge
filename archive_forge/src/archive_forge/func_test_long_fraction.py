import copy
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_long_fraction(self):
    inp = '\n        - 2011-10-02 16:45:00.1234      # expand with zeros\n        - 2011-10-02 16:45:00.123456\n        - 2011-10-02 16:45:00.12345612  # round to microseconds\n        - 2011-10-02 16:45:00.1234565   # round up\n        - 2011-10-02 16:45:00.12345678  # round up\n        '
    exp = '\n        - 2011-10-02 16:45:00.123400    # expand with zeros\n        - 2011-10-02 16:45:00.123456\n        - 2011-10-02 16:45:00.123456    # round to microseconds\n        - 2011-10-02 16:45:00.123457    # round up\n        - 2011-10-02 16:45:00.123457    # round up\n        '
    round_trip(inp, exp)