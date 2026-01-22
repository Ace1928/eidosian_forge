from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_221_sort_key(self):
    from srsly.ruamel_yaml import YAML
    from srsly.ruamel_yaml.compat import StringIO
    yaml = YAML()
    inp = dedent('        - four\n        - One    # 1\n        - Three  # 3\n        - five   # 5\n        - two    # 2\n        ')
    a = yaml.load(dedent(inp))
    a.sort(key=str.lower)
    buf = StringIO()
    yaml.dump(a, buf)
    exp = dedent('        - five   # 5\n        - four\n        - One    # 1\n        - Three  # 3\n        - two    # 2\n        ')
    assert buf.getvalue() == exp