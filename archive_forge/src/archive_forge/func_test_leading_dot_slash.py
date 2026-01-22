import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
def test_leading_dot_slash(self):
    self.assertMatch([('./foo', ['foo'], ['茶/foo', 'barfoo', 'x/y/foo']), ('./f*', ['foo'], ['foo/bar', 'foo/.bar', 'x/foo/y'])])