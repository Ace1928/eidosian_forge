import os
import sys
from unicodedata import normalize
from .. import osutils
from ..osutils import pathjoin
from . import TestCase, TestCaseWithTransport, TestSkipped
def test_functions(self):
    if osutils.normalizes_filenames():
        self.assertEqual(osutils.normalized_filename, osutils._accessible_normalized_filename)
    else:
        self.assertEqual(osutils.normalized_filename, osutils._inaccessible_normalized_filename)