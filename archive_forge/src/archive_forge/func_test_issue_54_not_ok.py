import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_issue_54_not_ok(self):
    yaml_str = dedent('        toplevel:\n\n            # some comment\n            sublevel: 300\n        ')
    d = round_trip_load(yaml_str)
    print(d.ca)
    y = round_trip_dump(d, indent=4)
    print(y.replace('\n', '$\n'))
    assert yaml_str == y