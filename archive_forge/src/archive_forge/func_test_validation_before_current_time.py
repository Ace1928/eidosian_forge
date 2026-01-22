from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
def test_validation_before_current_time(self):
    expiration = '1970-01-01'
    expected = 'Expiration %s is invalid: Expiration time is out of date.' % expiration
    self.assertFalse(self.constraint.validate(expiration, self.ctx))
    self.assertEqual(expected, str(self.constraint._error_message))