from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import server_groups as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import server_groups
def test_create_server_group_with_rules(self):
    kwargs = {'name': 'ig1', 'policy': 'anti-affinity', 'rules': {'max_server_per_host': 3}}
    server_group = self.cs.server_groups.create(**kwargs)
    self.assert_request_id(server_group, fakes.FAKE_REQUEST_ID_LIST)
    body = {'server_group': {'name': 'ig1', 'policy': 'anti-affinity', 'rules': {'max_server_per_host': 3}}}
    self.assert_called('POST', '/os-server-groups', body)
    self.assertIsInstance(server_group, server_groups.ServerGroup)