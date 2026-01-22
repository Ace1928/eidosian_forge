from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_236(self):
    inp = '\n        conf:\n          xx: {a: "b", c: []}\n          asd: "nn"\n        '
    d = round_trip(inp, preserve_quotes=True)