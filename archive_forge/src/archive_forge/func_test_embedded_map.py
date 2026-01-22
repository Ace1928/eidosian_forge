import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_embedded_map(self):
    round_trip('\n        - a: 1y\n          b: 2y\n\n          c: 3y\n        ')