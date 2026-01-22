import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_ctor_w_attrs_attrib_methods(self):
    from zope.interface.interface import Attribute
    from zope.interface.interface import fromFunction

    def _bar():
        """DOCSTRING"""
    ATTRS = {'foo': Attribute('Foo', ''), 'bar': fromFunction(_bar)}
    klass = self._getTargetClass()
    inst = klass('ITesting', attrs=ATTRS)
    self.assertEqual(inst.__name__, 'ITesting')
    self.assertEqual(inst.__doc__, '')
    self.assertEqual(inst.__bases__, ())
    self.assertEqual(inst.names(), ATTRS.keys())