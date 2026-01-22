from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_flavors_with_extra(self):
    self.use_compute_discovery()
    uris_to_mock = [dict(method='GET', uri='{endpoint}/flavors/detail?is_public=None'.format(endpoint=fakes.COMPUTE_ENDPOINT), json={'flavors': fakes.FAKE_FLAVOR_LIST})]
    uris_to_mock.extend([dict(method='GET', uri='{endpoint}/flavors/{id}/os-extra_specs'.format(endpoint=fakes.COMPUTE_ENDPOINT, id=flavor['id']), json={'extra_specs': {}}) for flavor in fakes.FAKE_FLAVOR_LIST])
    self.register_uris(uris_to_mock)
    flavors = self.cloud.list_flavors(get_extra=True)
    found = False
    for flavor in flavors:
        if flavor['name'] == 'vanilla':
            found = True
            break
    self.assertTrue(found)
    needed_keys = {'name', 'ram', 'vcpus', 'id', 'is_public', 'disk'}
    if found:
        self.assertTrue(needed_keys.issubset(flavor.keys()))
    self.assert_calls()