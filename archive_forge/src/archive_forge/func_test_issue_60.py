import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_issue_60(self):
    data = round_trip_load('\n        x: &base\n          a: 1\n        y:\n          <<: *base\n        ')
    assert data['x']['a'] == 1
    assert data['y']['a'] == 1
    if sys.version_info >= (3, 12):
        assert str(data['y']) == "ordereddict({'a': 1})"
    else:
        assert str(data['y']) == "ordereddict([('a', 1)])"