import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_class_conforms_to_IAdapterRegistration(self):
    from zope.interface.interfaces import IAdapterRegistration
    from zope.interface.verify import verifyClass
    verifyClass(IAdapterRegistration, self._getTargetClass())