import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, Undefined
def test_anytrait_static_notifiers_4(self):

    class AnytraitStaticNotifiers4(HasTraits):
        ok = Float

        def _anytrait_changed(self, name, old, new):
            if not hasattr(self, 'anycalls'):
                self.anycalls = []
            self.anycalls.append((name, old, new))
    obj = AnytraitStaticNotifiers4(ok=2)
    obj.ok = 3
    expected = [('trait_added', Undefined, 'anycalls'), ('ok', 0, 2), ('ok', 2, 3)]
    self.assertEqual(expected, obj.anycalls)