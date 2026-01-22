import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_existing_provides_is_not_spec(self):

    def foo():
        raise NotImplementedError()
    foo.__provides__ = object()
    spec = self._callFUT(foo)
    self.assertEqual(list(spec), [])