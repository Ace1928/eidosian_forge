from __future__ import print_function
import pytest  # NOQA
from .roundtrip import YAML  # does an automatic dedent on load
def test_rt_non_root_literal_scalar(self):
    yaml = YAML()
    s = 'testing123'
    ys = '\n        - |\n          {}\n        '
    ys = ys.format(s)
    d = yaml.load(ys)
    yaml.dump(d, compare=ys)