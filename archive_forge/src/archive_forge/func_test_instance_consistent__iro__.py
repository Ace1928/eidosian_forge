import unittest
def test_instance_consistent__iro__(self):
    from zope.interface import ro
    self.assertTrue(ro.is_consistent(self._getTargetInterface()))