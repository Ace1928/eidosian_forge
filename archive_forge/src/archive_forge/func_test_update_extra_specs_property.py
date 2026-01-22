from unittest import mock
from keystoneauth1 import adapter
from openstack.compute.v2 import flavor
from openstack.tests.unit import base
def test_update_extra_specs_property(self):
    sot = flavor.Flavor(**BASIC_EXAMPLE)
    resp = mock.Mock()
    resp.body = {'a': 'b'}
    resp.json = mock.Mock(return_value=resp.body)
    resp.status_code = 200
    self.sess.put = mock.Mock(return_value=resp)
    rsp = sot.update_extra_specs_property(self.sess, 'a', 'b')
    self.sess.put.assert_called_with('flavors/IDENTIFIER/os-extra_specs/a', json={'a': 'b'}, microversion=self.sess.default_microversion)
    self.assertEqual('b', rsp)