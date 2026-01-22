import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_does_not_leak_on_unique_classes(self):
    import gc
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    begin_count = len(gc.get_objects())
    for _ in range(1900):

        class TestClass:
            pass
        self._callFUT(TestClass, IFoo)
    gc.collect()
    end_count = len(gc.get_objects())
    fudge_factor = 0
    self.assertLessEqual(end_count, begin_count + fudge_factor)