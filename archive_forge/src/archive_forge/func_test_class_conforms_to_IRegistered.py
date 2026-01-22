import unittest
def test_class_conforms_to_IRegistered(self):
    from zope.interface.interfaces import IRegistered
    from zope.interface.verify import verifyClass
    verifyClass(IRegistered, self._getTargetClass())