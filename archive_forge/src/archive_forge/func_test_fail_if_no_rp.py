import uuid
from osc_placement.tests.functional import base
def test_fail_if_no_rp(self):
    self.assertCommandFailed(base.ARGUMENTS_MISSING, self.openstack, 'resource provider aggregate list')