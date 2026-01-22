from openstack.tests.functional import base
def test_qs(self):
    sot = self.conn.compute.get_quota_set(self.conn.current_project_id)
    self.assertIsNotNone(sot.key_pairs)