from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_221_sort_reverse(self):
    from srsly.ruamel_yaml import YAML
    from srsly.ruamel_yaml.compat import StringIO
    yaml = YAML()
    inp = dedent('        - d\n        - a  # 1\n        - c  # 3\n        - e  # 5\n        - b  # 2\n        ')
    a = yaml.load(dedent(inp))
    a.sort(reverse=True)
    buf = StringIO()
    yaml.dump(a, buf)
    exp = dedent('        - e  # 5\n        - d\n        - c  # 3\n        - b  # 2\n        - a  # 1\n        ')
    assert buf.getvalue() == exp