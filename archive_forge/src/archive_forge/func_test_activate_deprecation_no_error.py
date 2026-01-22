import warnings
from breezy import symbol_versioning
from breezy.symbol_versioning import (deprecated_function, deprecated_in,
from breezy.tests import TestCase
def test_activate_deprecation_no_error(self):
    symbol_versioning.activate_deprecation_warnings()
    self.assertFirstWarning('default', DeprecationWarning)