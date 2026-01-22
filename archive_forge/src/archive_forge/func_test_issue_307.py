from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_307(self):
    inp = '\n        %YAML 1.2\n        %TAG ! tag:example.com,2019/path#\n        ---\n        null\n        ...\n        '
    d = na_round_trip(inp)