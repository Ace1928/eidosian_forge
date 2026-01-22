from contextlib import contextmanager
import unittest
from traits.ctrait import CTrait
from traits.trait_converters import (
from traits.trait_factory import TraitFactory
from traits.trait_handlers import TraitCastType, TraitInstance
from traits.api import Any, Int
def test_trait_for_none(self):
    result = trait_for(None)
    self.assertIsInstance(result, CTrait)
    self.assertIsInstance(result.handler, TraitInstance)
    self.assertEqual(result.handler.aClass, type(None))