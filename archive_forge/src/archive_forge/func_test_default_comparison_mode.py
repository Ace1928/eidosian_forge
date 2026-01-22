import sys
import unittest.mock
import warnings
import weakref
from traits.api import HasTraits
from traits.constants import (
from traits.ctrait import CTrait
from traits.trait_errors import TraitError
from traits.trait_types import Any, Int, List
def test_default_comparison_mode(self):
    trait = CTrait(TraitKind.trait)
    self.assertIsInstance(trait.comparison_mode, ComparisonMode)
    self.assertEqual(trait.comparison_mode, ComparisonMode.equality)