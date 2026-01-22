import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
def test_mixed_slashes(self):
    """tests that multiple mixed slashes are collapsed to single forward
        slashes and trailing mixed slashes are removed"""
    self.assertEqual('/foo/bar', normalize_pattern('\\/\\foo//\\///bar/\\\\/'))