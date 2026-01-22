from __future__ import print_function
import pytest  # NOQA
from .roundtrip import YAML  # does an automatic dedent on load
def test_root_literal_scalar_no_indent_1_1_old_style(self):
    from textwrap import dedent
    from srsly.ruamel_yaml import safe_load
    s = 'testing123'
    inp = '\n        %YAML 1.1\n        --- |\n          {}\n        '
    d = safe_load(dedent(inp.format(s)))
    print(d)
    assert d == s + '\n'