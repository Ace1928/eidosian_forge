from openstack import exceptions
from openstack.shared_file_system.v2 import share as _share
from openstack.tests.functional.shared_file_system import base
def test_resize_share_smaller(self):
    smaller_size = 1
    self.user_cloud.share.resize_share(self.SHARE_ID, smaller_size)
    get_resized_share = self.user_cloud.share.get_share(self.SHARE_ID)
    self.user_cloud.share.wait_for_status(get_resized_share, status='available', failures=['error'], interval=5, wait=self._wait_for_timeout)
    self.assertEqual(smaller_size, get_resized_share.size)