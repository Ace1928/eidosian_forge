import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
def test_bad_pattern(self):
    """Ensure that globster handles bad patterns cleanly."""
    patterns = ['RE:[', '/home/foo', 'RE:*.cpp']
    g = Globster(patterns)
    e = self.assertRaises(lazy_regex.InvalidPattern, g.match, 'filename')
    self.assertContainsRe(e.msg, 'File.*ignore.*contains error.*RE:\\[.*RE:\\*\\.cpp', flags=re.DOTALL)