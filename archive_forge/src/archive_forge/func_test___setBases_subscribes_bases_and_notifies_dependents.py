import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___setBases_subscribes_bases_and_notifies_dependents(self):
    from zope.interface.interface import Interface
    spec = self._makeOne()
    dep = DummyDependent()
    spec.subscribe(dep)

    class I(Interface):
        pass

    class J(Interface):
        pass
    spec.__bases__ = (I,)
    self.assertEqual(dep._changed, [spec])
    self.assertEqual(I.dependents[spec], 1)
    spec.__bases__ = (J,)
    self.assertEqual(I.dependents.get(spec), None)
    self.assertEqual(J.dependents[spec], 1)