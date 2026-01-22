import warnings
from breezy import symbol_versioning
from breezy.symbol_versioning import (deprecated_function, deprecated_in,
from breezy.tests import TestCase
def test_deprecated_list(self):
    expected_warning = ("Modifying a_deprecated_list was deprecated in version 0.9.0. Don't use me", DeprecationWarning, 3)
    old_warning_method = symbol_versioning.warn
    try:
        symbol_versioning.set_warning_method(self.capture_warning)
        self.assertEqual(['one'], a_deprecated_list)
        self.assertEqual([], self._warnings)
        a_deprecated_list.append('foo')
        self.assertEqual([expected_warning], self._warnings)
        self.assertEqual(['one', 'foo'], a_deprecated_list)
        a_deprecated_list.extend(['bar', 'baz'])
        self.assertEqual([expected_warning] * 2, self._warnings)
        self.assertEqual(['one', 'foo', 'bar', 'baz'], a_deprecated_list)
        a_deprecated_list.insert(1, 'xxx')
        self.assertEqual([expected_warning] * 3, self._warnings)
        self.assertEqual(['one', 'xxx', 'foo', 'bar', 'baz'], a_deprecated_list)
        a_deprecated_list.remove('foo')
        self.assertEqual([expected_warning] * 4, self._warnings)
        self.assertEqual(['one', 'xxx', 'bar', 'baz'], a_deprecated_list)
        val = a_deprecated_list.pop()
        self.assertEqual([expected_warning] * 5, self._warnings)
        self.assertEqual('baz', val)
        self.assertEqual(['one', 'xxx', 'bar'], a_deprecated_list)
        val = a_deprecated_list.pop(1)
        self.assertEqual([expected_warning] * 6, self._warnings)
        self.assertEqual('xxx', val)
        self.assertEqual(['one', 'bar'], a_deprecated_list)
    finally:
        symbol_versioning.set_warning_method(old_warning_method)