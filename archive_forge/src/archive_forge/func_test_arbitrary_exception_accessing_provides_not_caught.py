import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_arbitrary_exception_accessing_provides_not_caught(self):

    class MyException(Exception):
        pass

    class Foo:
        __providedBy__ = self._makeOne()

        @property
        def __provides__(self):
            raise MyException
    foo = Foo()
    with self.assertRaises(MyException):
        getattr(foo, '__providedBy__')