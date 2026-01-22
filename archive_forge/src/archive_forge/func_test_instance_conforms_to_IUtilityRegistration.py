import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_instance_conforms_to_IUtilityRegistration(self):
    from zope.interface.interfaces import IUtilityRegistration
    from zope.interface.verify import verifyObject
    ur, _, _ = self._makeOne()
    verifyObject(IUtilityRegistration, ur)