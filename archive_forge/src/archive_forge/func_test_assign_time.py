import datetime
import unittest
from traits.testing.optional_dependencies import requires_traitsui, traitsui
from traits.api import HasStrictTraits, Time, TraitError
def test_assign_time(self):
    test_time = datetime.time(6, 3, 35)
    obj = HasTimeTraits()
    obj.simple_time = test_time
    self.assertEqual(obj.simple_time, test_time)