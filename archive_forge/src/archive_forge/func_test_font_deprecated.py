import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def test_font_deprecated(self):
    with self.assertWarnsRegex(DeprecationWarning, "'Font' in 'traits'"):
        Font()