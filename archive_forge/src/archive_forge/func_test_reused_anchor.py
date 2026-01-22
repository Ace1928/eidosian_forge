import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
def test_reused_anchor(self):
    from srsly.ruamel_yaml.error import ReusedAnchorWarning
    yaml = '\n        - &a\n          x: 1\n        - <<: *a\n        - &a\n          x: 2\n        - <<: *a\n        '
    with pytest.warns(ReusedAnchorWarning):
        data = round_trip(yaml)