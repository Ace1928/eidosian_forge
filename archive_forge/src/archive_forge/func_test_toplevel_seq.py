import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_toplevel_seq(self):
    round_trip('        - 1\n\n        - 2\n\n        - 3\n        ')