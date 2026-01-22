import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
def test_leading_asterisk_dot(self):
    self.assertMatch([('*.x', ['foo/bar/baz.x', '茶/Q.x', 'foo.y.x', '.foo.x', 'bar/.foo.x', '.x'], ['foo.x.y']), ('foo/*.bar', ['foo/b.bar', 'foo/a.b.bar', 'foo/.bar'], ['foo/bar']), ('*.~*', ['foo.py.~1~', '.foo.py.~1~'], [])])