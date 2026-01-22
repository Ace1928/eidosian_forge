import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, Undefined
def test_anytrait_static_notifiers_3_fail(self):
    obj = AnytraitStaticNotifiers3Fail()
    obj.fail = 1
    self.assertEqual(self.exceptions, [(obj, 'fail', 0, 1)])