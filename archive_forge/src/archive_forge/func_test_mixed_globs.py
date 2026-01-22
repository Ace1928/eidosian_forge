import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
def test_mixed_globs(self):
    """tests handling of combinations of path type matches.

        The types being extension, basename and full path.
        """
    patterns = ['*.foo', '.*.swp', './*.png']
    globster = Globster(patterns)
    self.assertEqual('*.foo', globster.match('bar.foo'))
    self.assertEqual('./*.png', globster.match('foo.png'))
    self.assertEqual(None, globster.match('foo/bar.png'))
    self.assertEqual('.*.swp', globster.match('foo/.bar.py.swp'))