from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_floating_ip_pools_exception(self):
    self.register_uris([dict(method='GET', uri='{endpoint}/extensions'.format(endpoint=fakes.COMPUTE_ENDPOINT), json={'extensions': [{'alias': 'os-floating-ip-pools', 'updated': '2014-12-03T00:00:00Z', 'name': 'FloatingIpPools', 'links': [], 'namespace': 'http://docs.openstack.org/compute/ext/fake_xml', 'description': 'Floating IPs support.'}]}), dict(method='GET', uri='{endpoint}/os-floating-ip-pools'.format(endpoint=fakes.COMPUTE_ENDPOINT), status_code=404)])
    self.assertRaises(exceptions.SDKException, self.cloud.list_floating_ip_pools)
    self.assert_calls()