import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_map_in_map_1(self):
    round_trip('\n        map1:\n          # comment 1\n          map2:\n            key1: val1\n        ')