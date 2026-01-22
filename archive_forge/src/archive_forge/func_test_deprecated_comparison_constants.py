import unittest
from traits.api import ComparisonMode
def test_deprecated_comparison_constants(self):
    from traits.api import NO_COMPARE, OBJECT_IDENTITY_COMPARE, RICH_COMPARE
    self.assertIs(NO_COMPARE, ComparisonMode.none)
    self.assertIs(OBJECT_IDENTITY_COMPARE, ComparisonMode.identity)
    self.assertIs(RICH_COMPARE, ComparisonMode.equality)