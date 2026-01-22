from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_inferred_default_microversion(self):
    self.cloud.config.config['compute_api_version'] = '2.42'
    server1 = fakes.make_fake_server('123', 'mickey')
    server2 = fakes.make_fake_server('345', 'mouse')
    self.register_uris([dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail']), request_headers={'OpenStack-API-Version': 'compute 2.42'}, json={'servers': [server1, server2]})])
    r = self.cloud.get_server('mickey', bare=True)
    self.assertIsNotNone(r)
    self.assertEqual(server1['name'], r['name'])
    self.assert_calls()