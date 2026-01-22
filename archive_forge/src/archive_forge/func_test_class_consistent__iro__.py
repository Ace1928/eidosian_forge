import unittest
def test_class_consistent__iro__(self):
    from zope.interface import implementedBy
    from zope.interface import ro
    self.assertTrue(ro.is_consistent(implementedBy(self._getTargetClass())))