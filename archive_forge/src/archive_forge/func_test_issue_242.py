from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_242(self):
    from srsly.ruamel_yaml.comments import CommentedMap
    d0 = CommentedMap([('a', 'b')])
    assert d0['a'] == 'b'