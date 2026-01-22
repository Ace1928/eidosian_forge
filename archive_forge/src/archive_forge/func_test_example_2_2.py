from .roundtrip import YAML
import pytest  # NOQA
@pytest.mark.xfail(strict=True)
def test_example_2_2():
    yaml = YAML()
    yaml.mapping_value_align = True
    yaml.round_trip('\n    hr:  65    # Home runs\n    avg: 0.278 # Batting average\n    rbi: 147   # Runs Batted In\n    ')