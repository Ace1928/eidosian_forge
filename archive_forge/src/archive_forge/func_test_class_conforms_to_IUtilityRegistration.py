import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_class_conforms_to_IUtilityRegistration(self):
    from zope.interface.interfaces import IUtilityRegistration
    from zope.interface.verify import verifyClass
    verifyClass(IUtilityRegistration, self._getTargetClass())