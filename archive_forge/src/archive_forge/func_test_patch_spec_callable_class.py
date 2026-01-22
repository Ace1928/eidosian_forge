import unittest2 as unittest
from mock.tests.support import is_instance, X, SomeClass
from mock import (
def test_patch_spec_callable_class(self):

    class CallableX(X):

        def __call__(self):
            pass

    class Sub(CallableX):
        pass

    class Multi(SomeClass, Sub):
        pass

    class OldStyle:

        def __call__(self):
            pass

    class OldStyleSub(OldStyle):
        pass
    for arg in ('spec', 'spec_set'):
        for Klass in (CallableX, Sub, Multi, OldStyle, OldStyleSub):
            with patch('%s.X' % __name__, **{arg: Klass}) as mock:
                instance = mock()
                mock.assert_called_once_with()
                self.assertTrue(is_instance(instance, MagicMock))
                self.assertRaises(AttributeError, getattr, instance, 'foobarbaz')
                result = instance()
                instance.assert_called_once_with()
                result(3, 2, 1)
                result.assert_called_once_with(3, 2, 1)
                result.foo(3, 2, 1)
                result.foo.assert_called_once_with(3, 2, 1)