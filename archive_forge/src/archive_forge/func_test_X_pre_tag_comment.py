import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, YAML
@pytest.mark.xfail(strict=True)
def test_X_pre_tag_comment(self):
    register_xxx()
    round_trip('        -\n          # hello\n          !xxx\n          name: Anthon\n          location: Germany\n          language: python\n        ')