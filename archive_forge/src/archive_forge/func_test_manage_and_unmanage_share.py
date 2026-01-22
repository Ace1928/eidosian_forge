from openstack import exceptions
from openstack.shared_file_system.v2 import share as _share
from openstack.tests.functional.shared_file_system import base
def test_manage_and_unmanage_share(self):
    self.operator_cloud.share.unmanage_share(self.SHARE_ID)
    self.operator_cloud.shared_file_system.wait_for_delete(self.NEW_SHARE, interval=2, wait=self._wait_for_timeout)
    try:
        self.operator_cloud.share.get_share(self.SHARE_ID)
    except exceptions.ResourceNotFound:
        pass
    managed_share = self.operator_cloud.share.manage_share(self.NEW_SHARE.share_protocol, self.export_path, self.share_host)
    self.operator_cloud.share.wait_for_status(managed_share, status='available', failures=['error'], interval=5, wait=self._wait_for_timeout)
    self.assertEqual(self.NEW_SHARE.share_protocol, managed_share.share_protocol)
    managed_host = self.operator_cloud.share.get_share(managed_share.id)['host']
    self.assertEqual(self.share_host, managed_host)