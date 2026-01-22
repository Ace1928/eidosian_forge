import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test__lt__NotImplemented(self):
    self.__check_NotImplemented_comparison('lt')