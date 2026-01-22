from .roundtrip import YAML
import pytest  # NOQA
@pytest.mark.xfail(strict=True, reason='case of NaN not preserved')
def test_example_2_20():
    yaml = YAML()
    yaml.round_trip('\n    canonical: 1.23015e+3\n    exponential: 12.3015e+02\n    fixed: 1230.15\n    negative infinity: -.inf\n    not a number: .NaN\n    ')