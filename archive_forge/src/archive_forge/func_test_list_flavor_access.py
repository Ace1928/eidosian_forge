from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_flavor_access(self):
    self.register_uris([dict(method='GET', uri='{endpoint}/flavors/vanilla/os-flavor-access'.format(endpoint=fakes.COMPUTE_ENDPOINT), json={'flavor_access': [{'flavor_id': 'vanilla', 'tenant_id': 'tenant_id'}]})])
    self.cloud.list_flavor_access('vanilla')
    self.assert_calls()