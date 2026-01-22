import contextlib
import gc
import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Str, WeakRef
from traits.testing.unittest_tools import UnittestTools
def test_target_freed_notification(self):
    eggs = Eggs(name='duck')
    spam = Spam(eggs=eggs)
    with self.assertTraitChanges(spam, 'eggs'):
        del eggs