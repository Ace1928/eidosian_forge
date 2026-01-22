from .roundtrip import YAML
import pytest  # NOQA
@pytest.mark.xfail(strict=True)
def test_example_2_15():
    yaml = YAML()
    yaml.round_trip('\n    >\n     Sammy Sosa completed another\n     fine season with great stats.\n\n       63 Home Runs\n       0.288 Batting Average\n\n     What a year!\n    ')