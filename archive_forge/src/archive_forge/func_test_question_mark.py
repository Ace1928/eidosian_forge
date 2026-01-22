import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
def test_question_mark(self):
    self.assertMatch([('?foo', ['xfoo', 'bar/xfoo', 'bar/茶foo', '.foo', 'bar/.foo'], ['bar/foo', 'foo']), ('foo?bar', ['fooxbar', 'foo.bar', 'foo茶bar', 'qyzzy/foo.bar'], ['foo/bar']), ('foo/?bar', ['foo/xbar', 'foo/茶bar', 'foo/.bar'], ['foo/bar', 'bar/foo/xbar'])])