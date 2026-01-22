import pytest
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
@pytest.mark.xfail(strict=True)
def test_standard_short_tag(self):
    round_trip('        !!map\n        name: Anthon\n        location: Germany\n        language: python\n        ')