import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_full_spectrum(self):

    class Foo:

        def bar(self, foo, bar='baz', *args, **kw):
            """DOCSTRING"""
    method = self._callFUT(Foo.bar)
    info = method.getSignatureInfo()
    self.assertEqual(list(info['positional']), ['foo', 'bar'])
    self.assertEqual(list(info['required']), ['foo'])
    self.assertEqual(info['optional'], {'bar': 'baz'})
    self.assertEqual(info['varargs'], 'args')
    self.assertEqual(info['kwargs'], 'kw')