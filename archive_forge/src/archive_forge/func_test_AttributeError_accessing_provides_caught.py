import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_AttributeError_accessing_provides_caught(self):

    class MyException(Exception):
        pass

    class Foo:
        __providedBy__ = self._makeOne()

        @property
        def __provides__(self):
            raise AttributeError
    foo = Foo()
    provided = getattr(foo, '__providedBy__')
    self.assertIsNotNone(provided)