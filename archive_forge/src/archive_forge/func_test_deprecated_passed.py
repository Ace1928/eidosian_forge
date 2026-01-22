import warnings
from breezy import symbol_versioning
from breezy.symbol_versioning import (deprecated_function, deprecated_in,
from breezy.tests import TestCase
def test_deprecated_passed(self):
    self.assertEqual(True, symbol_versioning.deprecated_passed(None))
    self.assertEqual(True, symbol_versioning.deprecated_passed(True))
    self.assertEqual(True, symbol_versioning.deprecated_passed(False))
    self.assertEqual(False, symbol_versioning.deprecated_passed(symbol_versioning.DEPRECATED_PARAMETER))