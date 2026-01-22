import pytest  # NOQA
from .roundtrip import dedent, round_trip, round_trip_load
def test_so_45681626(self):
    round_trip_load('{"in":{},"out":{}}')