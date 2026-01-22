from __future__ import print_function, absolute_import, division, unicode_literals
import pytest  # NOQA
from .roundtrip import dedent, round_trip_load, round_trip_dump
def test_calculate(self):
    s = dedent('        - 42\n        - 0b101010\n        - 0x_2a\n        - 0x2A\n        - 0o00_52\n        ')
    d = round_trip_load(s)
    for idx, elem in enumerate(d):
        elem -= 21
        d[idx] = elem
    for idx, elem in enumerate(d):
        elem *= 2
        d[idx] = elem
    for idx, elem in enumerate(d):
        t = elem
        elem **= 2
        elem //= t
        d[idx] = elem
    assert round_trip_dump(d) == s