from contextlib import contextmanager
import unittest
from traits.ctrait import CTrait
from traits.trait_converters import (
from traits.trait_factory import TraitFactory
from traits.trait_handlers import TraitCastType, TraitInstance
from traits.api import Any, Int
def test_as_ctrait_raise_exception(self):
    with self.assertRaises(TypeError):
        as_ctrait(1)
    with self.assertRaises(TypeError):
        as_ctrait(int)