from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_flavor_not_found(self):
    self.use_compute_discovery()
    self.register_uris([dict(method='GET', uri='{endpoint}/flavors/invalid'.format(endpoint=fakes.COMPUTE_ENDPOINT), status_code=404), dict(method='GET', uri='{endpoint}/flavors/detail?is_public=None'.format(endpoint=fakes.COMPUTE_ENDPOINT), json={'flavors': fakes.FAKE_FLAVOR_LIST})])
    self.assertFalse(self.cloud.delete_flavor('invalid'))
    self.assert_calls()