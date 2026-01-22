import sys
import unittest.mock
import warnings
import weakref
from traits.api import HasTraits
from traits.constants import (
from traits.ctrait import CTrait
from traits.trait_errors import TraitError
from traits.trait_types import Any, Int, List
def test_comparison_mode_enum(self):
    trait = CTrait(TraitKind.trait)
    trait.comparison_mode = ComparisonMode.none
    self.assertIsInstance(trait.comparison_mode, ComparisonMode)
    self.assertEqual(trait.comparison_mode, ComparisonMode.none)
    trait.comparison_mode = ComparisonMode.identity
    self.assertIsInstance(trait.comparison_mode, ComparisonMode)
    self.assertEqual(trait.comparison_mode, ComparisonMode.identity)
    trait.comparison_mode = ComparisonMode.equality
    self.assertIsInstance(trait.comparison_mode, ComparisonMode)
    self.assertEqual(trait.comparison_mode, ComparisonMode.equality)