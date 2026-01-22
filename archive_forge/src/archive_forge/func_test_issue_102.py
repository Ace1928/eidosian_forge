from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_102(self):
    yaml_str = dedent('\n        var1: #empty\n        var2: something #notempty\n        var3: {} #empty object\n        var4: {a: 1} #filled object\n        var5: [] #empty array\n        ')
    x = round_trip(yaml_str, preserve_quotes=True)