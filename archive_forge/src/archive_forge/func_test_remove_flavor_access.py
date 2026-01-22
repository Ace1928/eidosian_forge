from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_remove_flavor_access(self):
    self.register_uris([dict(method='POST', uri='{endpoint}/flavors/{id}/action'.format(endpoint=fakes.COMPUTE_ENDPOINT, id='flavor_id'), json={'flavor_access': []}, validate=dict(json={'removeTenantAccess': {'tenant': 'tenant_id'}}))])
    self.cloud.remove_flavor_access('flavor_id', 'tenant_id')
    self.assert_calls()