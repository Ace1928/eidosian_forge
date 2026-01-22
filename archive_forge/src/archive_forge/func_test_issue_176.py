from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_176(self):
    from srsly.ruamel_yaml import YAML
    yaml = YAML()
    seq = yaml.load('[1,2,3]')
    seq[:] = [1, 2, 3, 4]