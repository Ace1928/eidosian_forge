import unittest
def test_class_conforms(self):
    from zope.interface.verify import verifyClass
    verifyClass(self._getTargetInterface(), self._getTargetClass())