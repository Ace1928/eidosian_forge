import datetime
import unittest
from traits.testing.optional_dependencies import requires_traitsui, traitsui
from traits.api import Date, HasStrictTraits, TraitError
def test_assign_date(self):
    test_date = datetime.date(1975, 2, 13)
    obj = HasDateTraits()
    obj.simple_date = test_date
    self.assertEqual(obj.simple_date, test_date)