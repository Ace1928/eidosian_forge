import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_color_option(self):
    """Ensure options for color are valid.
        """
    out, err = self.run_bzr(['grep', '--color', 'foo', 'bar'], 3)
    self.assertEqual(out, '')
    self.assertContainsRe(err, 'Valid values for --color are', flags=TestGrep._reflags)