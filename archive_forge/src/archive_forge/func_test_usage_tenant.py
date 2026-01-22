import datetime
from novaclient.tests.functional import base
def test_usage_tenant(self):
    before = self._get_num_servers_by_tenant_from_usage_output()
    self._create_server()
    after = self._get_num_servers_by_tenant_from_usage_output()
    self.assertGreater(after, before)