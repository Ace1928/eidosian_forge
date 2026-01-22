from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_184(self):
    yaml_str = dedent('        test::test:\n          # test\n          foo:\n            bar: baz\n        ')
    d = round_trip_load(yaml_str)
    d['bar'] = 'foo'
    d.yaml_add_eol_comment('test1', 'bar')
    assert round_trip_dump(d) == yaml_str + 'bar: foo # test1\n'