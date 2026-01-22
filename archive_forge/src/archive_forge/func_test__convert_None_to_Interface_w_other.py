import unittest
from zope.interface.tests import OptimizationTestMixin
def test__convert_None_to_Interface_w_other(self):
    from zope.interface.adapter import _convert_None_to_Interface
    other = object()
    self.assertIs(_convert_None_to_Interface(other), other)