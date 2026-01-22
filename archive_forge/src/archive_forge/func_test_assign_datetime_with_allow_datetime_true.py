import datetime
import unittest
from traits.testing.optional_dependencies import requires_traitsui, traitsui
from traits.api import Date, HasStrictTraits, TraitError
def test_assign_datetime_with_allow_datetime_true(self):
    test_datetime = datetime.datetime(1975, 2, 13)
    obj = HasDateTraits()
    obj.datetime_allowed = test_datetime
    self.assertEqual(obj.datetime_allowed, test_datetime)