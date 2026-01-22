import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_issue_25a(self):
    round_trip('        - a: b\n          c: d\n          d:  # foo\n          - e: f\n        ')