import uuid
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_server_console_name_or_id(self):
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri='{endpoint}/servers/detail'.format(endpoint=fakes.COMPUTE_ENDPOINT), json={'servers': [self.server]}), dict(method='POST', uri='{endpoint}/servers/{id}/action'.format(endpoint=fakes.COMPUTE_ENDPOINT, id=self.server_id), json={'output': self.output}, validate=dict(json={'os-getConsoleOutput': {}}))])
    self.assertEqual(self.output, self.cloud.get_server_console(self.server['id']))
    self.assert_calls()