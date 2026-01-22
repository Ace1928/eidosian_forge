import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_register_unregister_nonequal_objects_provided(self):
    self.test_register_unregister_identical_objects_provided(identical=False)