from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_flavor_with_extra_specs(self):
    self.use_compute_discovery()
    flavor_uri = '{endpoint}/flavors/1'.format(endpoint=fakes.COMPUTE_ENDPOINT)
    flavor_extra_uri = '{endpoint}/flavors/1/os-extra_specs'.format(endpoint=fakes.COMPUTE_ENDPOINT)
    flavor_json = {'flavor': fakes.make_fake_flavor('1', 'vanilla')}
    flavor_extra_json = {'extra_specs': {'name': 'test'}}
    self.register_uris([dict(method='GET', uri=flavor_uri, json=flavor_json), dict(method='GET', uri=flavor_extra_uri, json=flavor_extra_json)])
    flavor1 = self.cloud.get_flavor_by_id('1', get_extra=True)
    self.assertEqual('1', flavor1['id'])
    self.assertEqual({'name': 'test'}, flavor1.extra_specs)
    flavor2 = self.cloud.get_flavor_by_id('1', get_extra=False)
    self.assertEqual('1', flavor2['id'])
    self.assertEqual({}, flavor2.extra_specs)