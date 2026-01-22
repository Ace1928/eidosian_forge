import pytest
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
@pytest.mark.xfail(strict=True)
def test_comment_dash_line(self):
    round_trip('\n        - # abc\n           a: 1\n           b: 2\n        ')