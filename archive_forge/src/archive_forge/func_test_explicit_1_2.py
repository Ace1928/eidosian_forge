import pytest  # NOQA
from .roundtrip import dedent, round_trip, round_trip_load
def test_explicit_1_2(self):
    r = load('        %YAML 1.2\n        ---\n        - 12:34:56\n        - 012\n        - 012345678\n        - 0o12\n        - on\n        - off\n        - yes\n        - no\n        - true\n        ')
    assert r[0] == '12:34:56'
    assert r[1] == 12
    assert r[2] == 12345678
    assert r[3] == 10
    assert r[4] == 'on'
    assert r[5] == 'off'
    assert r[6] == 'yes'
    assert r[7] == 'no'
    assert r[8] is True