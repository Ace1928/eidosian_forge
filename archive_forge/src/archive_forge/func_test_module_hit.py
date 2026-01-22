import unittest
def test_module_hit(self):
    from zope.interface.tests import dummy
    from zope.interface.tests.idummy import IDummyModule
    self._callFUT(IDummyModule, dummy)