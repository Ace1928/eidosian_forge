from unittest import mock
from keystoneauth1 import adapter
from openstack.compute.v2 import flavor
from openstack.tests.unit import base
def test_delete_extra_specs_property(self):
    sot = flavor.Flavor(**BASIC_EXAMPLE)
    resp = mock.Mock()
    resp.body = None
    resp.json = mock.Mock(return_value=resp.body)
    resp.status_code = 200
    self.sess.delete = mock.Mock(return_value=resp)
    rsp = sot.delete_extra_specs_property(self.sess, 'a')
    self.sess.delete.assert_called_with('flavors/IDENTIFIER/os-extra_specs/a', microversion=self.sess.default_microversion)
    self.assertIsNone(rsp)