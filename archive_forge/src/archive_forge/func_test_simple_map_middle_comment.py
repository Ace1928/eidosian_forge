import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_simple_map_middle_comment(self):
    round_trip('\n        abc: 1\n        # C 3a\n        # C 3b\n        ghi: 2\n        ')