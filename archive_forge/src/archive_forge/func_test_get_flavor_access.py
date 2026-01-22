from unittest import mock
from keystoneauth1 import adapter
from openstack.compute.v2 import flavor
from openstack.tests.unit import base
def test_get_flavor_access(self):
    sot = flavor.Flavor(**BASIC_EXAMPLE)
    resp = mock.Mock()
    resp.body = {'flavor_access': [{'flavor_id': 'fake_flavor', 'tenant_id': 'fake_tenant'}]}
    resp.json = mock.Mock(return_value=resp.body)
    resp.status_code = 200
    self.sess.get = mock.Mock(return_value=resp)
    rsp = sot.get_access(self.sess)
    self.sess.get.assert_called_with('flavors/IDENTIFIER/os-flavor-access')
    self.assertEqual(resp.body['flavor_access'], rsp)