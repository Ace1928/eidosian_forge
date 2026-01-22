import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_ctor_attrs_w_invalid_attr_type(self):
    from zope.interface.exceptions import InvalidInterface
    ATTRS = {'invalid': object()}
    klass = self._getTargetClass()
    self.assertRaises(InvalidInterface, klass, 'ITesting', attrs=ATTRS)