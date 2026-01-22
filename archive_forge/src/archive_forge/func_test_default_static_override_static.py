import pickle
import unittest
from traits.api import HasTraits, Int, PrefixMap, TraitError
def test_default_static_override_static(self):

    class BasePerson(HasTraits):
        married = PrefixMap({'yes': 1, 'yeah': 1, 'no': 0, 'nah': 0}, default_value='nah')

    class Person(BasePerson):
        married = 'yes'
    p = Person()
    self.assertEqual(p.married, 'yes')
    self.assertEqual(p.married_, 1)