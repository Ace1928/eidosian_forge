import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, YAML
def test_encoded_unicode_tag(self):
    round_trip_load("\n        s: !!python/%75nicode 'abc'\n        ")