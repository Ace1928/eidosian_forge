import pickle
import unittest
from traits.api import HasTraits, TraitError, PrefixList
def test_default_subject_to_validation(self):
    with self.assertRaises(ValueError):

        class A(HasTraits):
            foo = PrefixList(['zero', 'one', 'two'], default_value='uno')