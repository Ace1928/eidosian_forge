from __future__ import print_function
import pytest  # NOQA
from .roundtrip import YAML  # does an automatic dedent on load
def test_rt_root_literal_scalar_indent(self):
    yaml = YAML()
    yaml.explicit_start = True
    yaml.indent = 4
    s = 'testing123'
    ys = '\n        --- |\n            {}\n        '
    ys = ys.format(s)
    d = yaml.load(ys)
    yaml.dump(d, compare=ys)