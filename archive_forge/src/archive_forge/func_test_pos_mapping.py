import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_pos_mapping(self):
    data = load('\n        a: 1\n        b: 2\n        c: 3\n        # comment\n        klm: 42\n        d: 4\n        ')
    assert data.lc.key('klm') == (4, 0)
    assert data.lc.value('klm') == (4, 5)