from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import server_groups as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import server_groups
def test_list_server_groups_with_all_projects(self):
    result = self.cs.server_groups.list(all_projects=True)
    self.assert_request_id(result, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('GET', '/os-server-groups?all_projects=True')
    self.assertEqual(8, len(result))
    for server_group in result:
        self.assertIsInstance(server_group, server_groups.ServerGroup)