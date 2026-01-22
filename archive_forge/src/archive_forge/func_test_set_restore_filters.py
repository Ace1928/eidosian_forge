import warnings
from breezy import symbol_versioning
from breezy.symbol_versioning import (deprecated_function, deprecated_in,
from breezy.tests import TestCase
def test_set_restore_filters(self):
    original_filters = warnings.filters[:]
    symbol_versioning.suppress_deprecation_warnings()()
    self.assertEqual(original_filters, warnings.filters)