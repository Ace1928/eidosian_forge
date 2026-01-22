import unittest
def test_instance_conforms_to_IUnregistered(self):
    from zope.interface.interfaces import IUnregistered
    from zope.interface.verify import verifyObject
    verifyObject(IUnregistered, self._makeOne())