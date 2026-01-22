from .roundtrip import YAML
import pytest  # NOQA
@pytest.mark.xfail(strict=True)
def test_example_2_12():
    yaml = YAML()
    yaml.explicit_start = True
    yaml.round_trip('\n    ---\n    # Products purchased\n    - item    : Super Hoop\n      quantity: 1\n    - item    : Basketball\n      quantity: 4\n    - item    : Big Shoes\n      quantity: 1\n    ')