from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_233a(self):
    from srsly.ruamel_yaml import YAML
    import json
    yaml = YAML()
    data = yaml.load('[]')
    json_str = json.dumps(data)