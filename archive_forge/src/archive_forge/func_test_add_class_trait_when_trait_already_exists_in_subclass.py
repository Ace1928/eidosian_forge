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
def test_add_class_trait_when_trait_already_exists_in_subclass(self):

    class A(HasTraits):
        pass

    class B(A):
        foo = Int()
    A.add_class_trait('foo', Str())
    self.assertEqual(A().foo, '')
    self.assertEqual(B().foo, 0)