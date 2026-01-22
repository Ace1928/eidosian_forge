import unittest
from zope.interface.tests import OptimizationTestMixin
def test__convert_None_to_Interface_w_None(self):
    from zope.interface.adapter import _convert_None_to_Interface
    from zope.interface.interface import Interface
    self.assertIs(_convert_None_to_Interface(None), Interface)