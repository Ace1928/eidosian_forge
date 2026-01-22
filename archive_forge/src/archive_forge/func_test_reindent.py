import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_reindent(self):
    x = '        a:\n          b:     # comment 1\n            c: 1 # comment 2\n        '
    d = round_trip_load(x)
    y = round_trip_dump(d, indent=4)
    assert y == dedent('        a:\n            b:   # comment 1\n                c: 1 # comment 2\n        ')