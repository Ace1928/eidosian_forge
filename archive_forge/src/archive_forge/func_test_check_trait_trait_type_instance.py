from contextlib import contextmanager
import unittest
from traits.ctrait import CTrait
from traits.trait_converters import (
from traits.trait_factory import TraitFactory
from traits.trait_handlers import TraitCastType, TraitInstance
from traits.api import Any, Int
def test_check_trait_trait_type_instance(self):
    trait = Int()
    result = check_trait(trait)
    self.assertIsInstance(result, CTrait)
    self.assertIs(result.handler, trait)