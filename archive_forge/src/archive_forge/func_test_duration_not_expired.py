from testtools import matchers
from heat.common import timeutils as util
from heat.tests import common
def test_duration_not_expired(self):
    self.assertFalse(util.Duration(1.0).expired())