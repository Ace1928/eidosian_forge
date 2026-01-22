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
def test_add_class_trait_when_trait_already_exists(self):

    class A(HasTraits):
        foo = Int()
    with self.assertRaises(TraitError):
        A.add_class_trait('foo', List())
    self.assertEqual(A().foo, 0)
    with self.assertRaises(AttributeError):
        A().foo_items