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
def test_add_class_trait_affects_subclasses(self):

    class A(HasTraits):
        pass

    class B(A):
        pass

    class C(B):
        pass

    class D(B):
        pass
    A.add_class_trait('y', Str())
    self.assertEqual(A().y, '')
    self.assertEqual(B().y, '')
    self.assertEqual(C().y, '')
    self.assertEqual(D().y, '')