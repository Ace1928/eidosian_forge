from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_300a(self):
    import srsly.ruamel_yaml
    inp = dedent('\n        %YAML 1.1\n        %TAG ! tag:example.com,2019/path#fragment\n        ---\n        null\n        ')
    yaml = YAML()
    with pytest.raises(srsly.ruamel_yaml.scanner.ScannerError, match='while scanning a directive'):
        yaml.load(inp)