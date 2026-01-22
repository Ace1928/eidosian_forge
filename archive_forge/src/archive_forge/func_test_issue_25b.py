import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_issue_25b(self):
    round_trip('        var1: #empty\n        var2: something #notempty\n        ')