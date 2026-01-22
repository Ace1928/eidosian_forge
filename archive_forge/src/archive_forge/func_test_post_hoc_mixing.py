import abc
import unittest
import warnings
from traits.api import ABCHasTraits, ABCMetaHasTraits, HasTraits, Int, Float
def test_post_hoc_mixing(self):

    class TraitedBar(HasTraits, AbstractBar, metaclass=ABCMetaHasTraits):
        x = Int(10)

        def bar(self):
            return 'bar'
    traited = TraitedBar()
    self.assertTrue(isinstance(traited, AbstractBar))
    self.assertEqual(traited.x, 10)