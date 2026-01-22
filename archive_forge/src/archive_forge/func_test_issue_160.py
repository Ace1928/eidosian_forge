from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_160(self):
    from srsly.ruamel_yaml.compat import StringIO
    s = dedent('        root:\n            # a comment\n            - {some_key: "value"}\n\n        foo: 32\n        bar: 32\n        ')
    a = round_trip_load(s)
    del a['root'][0]['some_key']
    buf = StringIO()
    round_trip_dump(a, buf, block_seq_indent=4)
    exp = dedent('        root:\n            # a comment\n            - {}\n\n        foo: 32\n        bar: 32\n        ')
    assert buf.getvalue() == exp