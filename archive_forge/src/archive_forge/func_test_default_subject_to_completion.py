import pickle
import unittest
from traits.api import HasTraits, TraitError, PrefixList
def test_default_subject_to_completion(self):

    class A(HasTraits):
        foo = PrefixList(['zero', 'one', 'two'], default_value='o')
    a = A()
    self.assertEqual(a.foo, 'one')