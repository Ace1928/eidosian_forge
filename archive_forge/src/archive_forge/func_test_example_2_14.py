from .roundtrip import YAML
import pytest  # NOQA
@pytest.mark.xfail(strict=True)
def test_example_2_14():
    yaml = YAML()
    yaml.explicit_start = True
    yaml.indent(root_scalar=2)
    yaml.round_trip("\n    --- >\n      Mark McGwire's\n      year was crippled\n      by a knee injury.\n    ")