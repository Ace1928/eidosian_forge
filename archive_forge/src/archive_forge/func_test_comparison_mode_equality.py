import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def test_comparison_mode_equality(self):

    class HasComparisonMode(HasTraits):
        bar = Trait(comparison_mode=ComparisonMode.equality)
    old_compare = HasComparisonMode()
    events = []
    old_compare.on_trait_change(lambda: events.append(None), 'bar')
    some_list = [1, 2, 3]
    self.assertEqual(len(events), 0)
    old_compare.bar = some_list
    self.assertEqual(len(events), 1)
    old_compare.bar = some_list
    self.assertEqual(len(events), 1)
    old_compare.bar = [1, 2, 3]
    self.assertEqual(len(events), 1)
    old_compare.bar = [4, 5, 6]
    self.assertEqual(len(events), 2)