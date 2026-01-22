from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_300(self):
    from srsly.ruamel_yaml import YAML
    inp = dedent('\n        %YAML 1.2\n        %TAG ! tag:example.com,2019/path#fragment\n        ---\n        null\n        ')
    YAML().load(inp)