import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, YAML
def test_sequence2(self):
    yaml = YAML()
    yaml.mapping_value_align = True
    yaml.round_trip('\n        - !Sequence [a, b: 1, c: {d: 3}]\n        ')