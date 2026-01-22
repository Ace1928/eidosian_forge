from __future__ import print_function
import pytest  # NOQA
from .roundtrip import YAML  # does an automatic dedent on load
def test_root_literal_scalar_indent_offset_four(self):
    yaml = YAML()
    s = 'testing123'
    inp = '\n        --- |4\n            {}\n        '
    d = yaml.load(inp.format(s))
    print(d)
    assert d == s + '\n'