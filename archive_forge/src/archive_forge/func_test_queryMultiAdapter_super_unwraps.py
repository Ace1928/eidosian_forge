import unittest
from zope.interface.tests import OptimizationTestMixin
def test_queryMultiAdapter_super_unwraps(self):
    alb = self._makeOne(self._makeRegistry())

    def lookup(*args):
        return factory

    def factory(*args):
        return args
    alb.lookup = lookup
    objects = [super(), 42, 'abc', super()]
    result = alb.queryMultiAdapter(objects, None)
    self.assertEqual(result, (self, 42, 'abc', self))