import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_w_single(self):
    from zope.interface.interface import TAGGED_DATA
    from zope.interface.interface import taggedValue

    class Foo:
        taggedValue('bar', ['baz'])
    self.assertEqual(getattr(Foo, TAGGED_DATA, None), {'bar': ['baz']})