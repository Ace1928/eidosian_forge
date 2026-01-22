import sys
import unittest.mock
import warnings
import weakref
from traits.api import HasTraits
from traits.constants import (
from traits.ctrait import CTrait
from traits.trait_errors import TraitError
from traits.trait_types import Any, Int, List
def test_unsafe_set_value(self):

    def get_handler_refcount():
        sys.getrefcount(tr.handler)
    weakrefable_object = {1, 2, 3}
    tr = CTrait(0)
    tr.handler = Any(weakrefable_object)
    finalizer = weakref.finalize(weakrefable_object, get_handler_refcount)
    del weakrefable_object
    self.assertTrue(finalizer.alive)
    tr.handler = None
    self.assertFalse(finalizer.alive)
    self.assertIsNone(tr.handler)