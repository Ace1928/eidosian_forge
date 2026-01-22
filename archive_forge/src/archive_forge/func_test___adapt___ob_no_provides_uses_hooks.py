import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___adapt___ob_no_provides_uses_hooks(self):
    from zope.interface import interface
    ib = self._makeOne(False)
    adapted = object()
    _missed = []

    def _hook_miss(iface, obj):
        _missed.append((iface, obj))

    def _hook_hit(iface, obj):
        return obj
    with _Monkey(interface, adapter_hooks=[_hook_miss, _hook_hit]):
        self.assertIs(ib.__adapt__(adapted), adapted)
        self.assertEqual(_missed, [(ib, adapted)])