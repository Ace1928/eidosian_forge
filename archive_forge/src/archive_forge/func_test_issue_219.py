from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_219(self):
    yaml_str = dedent('        [StackName: AWS::StackName]\n        ')
    d = round_trip_load(yaml_str)