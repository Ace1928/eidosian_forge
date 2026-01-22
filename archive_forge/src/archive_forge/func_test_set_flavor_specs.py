from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_set_flavor_specs(self):
    self.use_compute_discovery()
    extra_specs = dict(key1='value1')
    self.register_uris([dict(method='POST', uri='{endpoint}/flavors/{id}/os-extra_specs'.format(endpoint=fakes.COMPUTE_ENDPOINT, id=1), json=dict(extra_specs=extra_specs))])
    self.cloud.set_flavor_specs(1, extra_specs)
    self.assert_calls()