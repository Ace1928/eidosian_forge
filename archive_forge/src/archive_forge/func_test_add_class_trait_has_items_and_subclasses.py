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
def test_add_class_trait_has_items_and_subclasses(self):

    class A(HasTraits):
        pass

    class B(A):
        pass

    class C(B):
        pass
    A.add_class_trait('x', List(Int))
    self.assertEqual(A().x, [])
    self.assertEqual(B().x, [])
    self.assertEqual(C().x, [])
    A.add_class_trait('y', Map({'yes': 1, 'no': 0}, default_value='no'))
    self.assertEqual(A().y, 'no')
    self.assertEqual(B().y, 'no')
    self.assertEqual(C().y, 'no')