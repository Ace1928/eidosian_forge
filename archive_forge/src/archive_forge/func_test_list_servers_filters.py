import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_servers_filters(self):
    """This test verifies that when list_servers is called with
        `filters` dict that it passes it to nova."""
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail'], qs_elements=['deleted=True', 'changes-since=2014-12-03T00:00:00Z']), complete_qs=True, json={'servers': []})])
    self.cloud.list_servers(filters={'deleted': True, 'changes-since': '2014-12-03T00:00:00Z'})
    self.assert_calls()