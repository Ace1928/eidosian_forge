from openstack.network.v2 import network_segment_range
from openstack.tests.functional import base
def test_create_delete(self):
    del_test_seg_range = self.operator_cloud.network.delete_network_segment_range(self.NETWORK_SEGMENT_RANGE_ID)
    self.assertIsNone(del_test_seg_range)