import copy
import pickle
import unittest
from traits.has_traits import (
from traits.ctrait import CTrait
from traits.observation.api import (
from traits.observation.exception_handling import (
from traits.traits import ForwardProperty, generic_trait
from traits.trait_types import Event, Float, Instance, Int, List, Map, Str
from traits.trait_errors import TraitError
def test__instance_traits(self):

    class Base(HasTraits):
        pin = Int
    a = Base()
    a_instance_traits = a._instance_traits()
    self.assertIsInstance(a_instance_traits, dict)
    self.assertIs(a._instance_traits(), a_instance_traits)
    b = Base()
    self.assertIsNot(b._instance_traits(), a_instance_traits)