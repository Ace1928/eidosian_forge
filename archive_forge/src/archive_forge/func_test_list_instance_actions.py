from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import instance_action
def test_list_instance_actions(self):
    server_uuid = '1234'
    ial = self.cs.instance_action.list(server_uuid)
    self.assert_request_id(ial, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('GET', '/servers/%s/os-instance-actions' % server_uuid)