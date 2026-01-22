import datetime
import unittest
from traits.testing.optional_dependencies import requires_traitsui, traitsui
from traits.api import Datetime, HasStrictTraits, TraitError
def test_assign_non_datetime(self):
    obj = HasDatetimeTraits()
    with self.assertRaises(TraitError) as exception_context:
        obj.simple_datetime = '2021-02-05 12:00:00'
    message = str(exception_context.exception)
    self.assertIn('must be a datetime or None, but', message)