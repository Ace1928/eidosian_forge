from openstack.shared_file_system.v2 import share_group as _share_group
from openstack.tests.functional.shared_file_system import base
def test_list_delete_share_group(self):
    s_grps = self.user_cloud.shared_file_system.share_groups()
    self.assertGreater(len(list(s_grps)), 0)
    for s_grp in s_grps:
        for attribute in ('id', 'name', 'created_at'):
            self.assertTrue(hasattr(s_grp, attribute))
        sot = self.conn.shared_file_system.delete_share_group(s_grp)
        self.assertIsNone(sot)