import warnings
from breezy import symbol_versioning
from breezy.symbol_versioning import (deprecated_function, deprecated_in,
from breezy.tests import TestCase
def check_deprecated_callable(self, expected_warning, expected_docstring, expected_name, expected_module, deprecated_callable):
    if __doc__ is None:
        expected_docstring = expected_docstring.split('\n')[-2].lstrip()
    old_warning_method = symbol_versioning.warn
    try:
        symbol_versioning.set_warning_method(self.capture_warning)
        self.assertEqual(1, deprecated_callable())
        self.assertEqual([expected_warning], self._warnings)
        deprecated_callable()
        self.assertEqual([expected_warning, expected_warning], self._warnings)
        self.assertEqualDiff(expected_docstring, deprecated_callable.__doc__)
        self.assertEqualDiff(expected_name, deprecated_callable.__name__)
        self.assertEqualDiff(expected_module, deprecated_callable.__module__)
        self.assertTrue(deprecated_callable.is_deprecated)
    finally:
        symbol_versioning.set_warning_method(old_warning_method)