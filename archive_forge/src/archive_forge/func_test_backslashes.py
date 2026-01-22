import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
def test_backslashes(self):
    """tests that backslashes are converted to forward slashes, multiple
        backslashes are collapsed to single forward slashes and trailing
        backslashes are removed"""
    self.assertEqual('/', normalize_pattern('\\'))
    self.assertEqual('/', normalize_pattern('\\\\'))
    self.assertEqual('/foo/bar', normalize_pattern('\\foo\\bar'))
    self.assertEqual('foo/bar', normalize_pattern('foo\\bar\\'))
    self.assertEqual('/foo/bar', normalize_pattern('\\\\foo\\\\bar\\\\'))