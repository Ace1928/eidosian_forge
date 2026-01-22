from openstack.tests.functional import base
def test_set_quotas(self):
    """Test set quotas functionality"""
    if not self.operator_cloud:
        self.skipTest('Operator cloud is required for this test')
    quotas = self.operator_cloud.get_volume_quotas('demo')
    volumes = quotas['volumes']
    self.operator_cloud.set_volume_quotas('demo', volumes=volumes + 1)
    self.assertEqual(volumes + 1, self.operator_cloud.get_volume_quotas('demo')['volumes'])
    self.operator_cloud.delete_volume_quotas('demo')
    self.assertEqual(volumes, self.operator_cloud.get_volume_quotas('demo')['volumes'])