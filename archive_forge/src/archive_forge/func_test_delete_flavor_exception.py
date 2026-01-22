from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_flavor_exception(self):
    self.use_compute_discovery()
    self.register_uris([dict(method='GET', uri='{endpoint}/flavors/vanilla'.format(endpoint=fakes.COMPUTE_ENDPOINT), json=fakes.FAKE_FLAVOR), dict(method='GET', uri='{endpoint}/flavors/detail?is_public=None'.format(endpoint=fakes.COMPUTE_ENDPOINT), json={'flavors': fakes.FAKE_FLAVOR_LIST}), dict(method='DELETE', uri='{endpoint}/flavors/{id}'.format(endpoint=fakes.COMPUTE_ENDPOINT, id=fakes.FLAVOR_ID), status_code=503)])
    self.assertRaises(exceptions.SDKException, self.cloud.delete_flavor, 'vanilla')