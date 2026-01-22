import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_round_trip_ordering(self):
    round_trip('\n        a: 1\n        b: 2\n        c: 3\n        b1: 2\n        b2: 2\n        d: 4\n        e: 5\n        f: 6\n        ')