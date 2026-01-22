import unittest
def test_module_miss(self):
    from zope.interface import Interface
    from zope.interface.exceptions import DoesNotImplement
    from zope.interface.tests import dummy

    class IDummyModule(Interface):
        pass
    self.assertRaises(DoesNotImplement, self._callFUT, IDummyModule, dummy)