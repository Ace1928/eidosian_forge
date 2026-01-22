from openstack.compute.v2 import limits as _limits
from openstack.tests.functional import base
def test_get_other_compute_limits(self):
    """Test quotas functionality"""
    if not self.operator_cloud:
        self.skipTest('Operator cloud is required for this test')
    limits = self.operator_cloud.get_compute_limits('demo')
    self.assertIsNotNone(limits)
    self.assertTrue(hasattr(limits, 'server_meta'))
    self.assertFalse(hasattr(limits, 'maxImageMeta'))