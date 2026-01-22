import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_verifyClass(self):
    from zope.interface import Attribute
    from zope.interface import Interface
    from zope.interface.verify import verifyClass

    class ICheckMe(Interface):
        attr = Attribute('My attr')

        def method():
            """A method"""

    class CheckMe:
        __implemented__ = ICheckMe
        attr = 'value'

        def method(self):
            raise NotImplementedError()
    self.assertTrue(verifyClass(ICheckMe, CheckMe))