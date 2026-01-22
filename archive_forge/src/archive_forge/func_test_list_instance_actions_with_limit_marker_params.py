from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import instance_action
def test_list_instance_actions_with_limit_marker_params(self):
    server_uuid = '1234'
    marker = '12140183-c814-4ddf-8453-6df43028aa67'
    ias = self.cs.instance_action.list(server_uuid, marker=marker, limit=10, changes_since='2016-02-29T06:23:22')
    self.assert_request_id(ias, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('GET', '/servers/%s/os-instance-actions?changes-since=%s&limit=10&marker=%s' % (server_uuid, '2016-02-29T06%3A23%3A22', marker))
    for ia in ias:
        self.assertIsInstance(ia, instance_action.InstanceAction)