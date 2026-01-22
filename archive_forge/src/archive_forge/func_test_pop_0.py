import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_pop_0(self):
    d = round_trip_load(self.ins)
    d['ab'].pop(0)
    y = round_trip_dump(d, indent=2)
    print(y)
    assert y == dedent('        ab:\n        - b      # b\n        - c\n        - d      # d\n\n        de:\n        - 1\n        - 2\n        ')