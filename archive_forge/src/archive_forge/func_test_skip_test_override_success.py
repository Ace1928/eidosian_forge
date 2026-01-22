import sys
import warnings
from oslo_log import log
from sqlalchemy import exc
from testtools import matchers
from keystone.tests import unit
def test_skip_test_override_success(self):
    test = self.TestChild('test_in_parent')
    result = test.run()
    self.assertEqual([], result.decorated.errors)