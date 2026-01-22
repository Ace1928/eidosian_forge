from openstack.compute.v2 import limits as _limits
from openstack.tests.functional import base
def test_get_our_compute_limits(self):
    """Test quotas functionality"""
    limits = self.user_cloud.get_compute_limits()
    self.assertIsNotNone(limits)
    self.assertIsInstance(limits, _limits.AbsoluteLimits)
    self.assertIsNotNone(limits.server_meta)
    self.assertIsNotNone(limits.image_meta)