import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_issue_25_01(self):
    round_trip('        a:        # comment 1\n                  #  comment 2\n        - b:      #   comment 3\n            c: 1  #    comment 4\n        ')