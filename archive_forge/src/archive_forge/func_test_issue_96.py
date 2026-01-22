import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_issue_96(self):
    round_trip('        a:\n          b:\n            c: c_val\n            d:\n\n          e:\n            g: g_val\n        ')