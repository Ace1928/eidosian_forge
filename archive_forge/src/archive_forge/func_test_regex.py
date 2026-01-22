import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
def test_regex(self):
    self.assertMatch([('RE:(a|b|c+)', ['a', 'b', 'ccc'], ['d', 'aa', 'c+', '-a']), ('RE:(?:a|b|c+)', ['a', 'b', 'ccc'], ['d', 'aa', 'c+', '-a']), ('RE:(?P<a>.)(?P=a)', ['a'], ['ab', 'aa', 'aaa']), ('RE:a\\\\\\', ['a\\'], ['a', 'ab', 'aa', 'aaa'])])