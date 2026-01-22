from .roundtrip import YAML
import pytest  # NOQA
@pytest.mark.xfail(strict=True)
def test_example_2_11():
    yaml = YAML()
    yaml.round_trip('\n    ? - Detroit Tigers\n      - Chicago cubs\n    :\n      - 2001-07-23\n\n    ? [ New York Yankees,\n        Atlanta Braves ]\n    : [ 2001-07-02, 2001-08-12,\n        2001-08-14 ]\n    ')