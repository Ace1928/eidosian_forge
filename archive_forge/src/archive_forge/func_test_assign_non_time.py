import datetime
import unittest
from traits.testing.optional_dependencies import requires_traitsui, traitsui
from traits.api import HasStrictTraits, Time, TraitError
def test_assign_non_time(self):
    obj = HasTimeTraits()
    with self.assertRaises(TraitError) as exception_context:
        obj.simple_time = '12:00:00'
    message = str(exception_context.exception)
    self.assertIn('must be a time or None, but', message)