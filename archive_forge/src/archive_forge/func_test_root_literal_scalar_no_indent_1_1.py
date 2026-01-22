from __future__ import print_function
import pytest  # NOQA
from .roundtrip import YAML  # does an automatic dedent on load
def test_root_literal_scalar_no_indent_1_1(self):
    yaml = YAML()
    s = 'testing123'
    inp = '\n        %YAML 1.1\n        --- |\n        {}\n        '
    d = yaml.load(inp.format(s))
    print(d)
    assert d == s + '\n'