from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import group_type
from openstack.tests.unit import base
def test_get_group_specs_property(self):
    sot = group_type.GroupType(**GROUP_TYPE)
    resp = mock.Mock()
    resp.body = {'a': 'b'}
    resp.json = mock.Mock(return_value=resp.body)
    resp.status_code = 200
    self.sess.get = mock.Mock(return_value=resp)
    rsp = sot.get_group_specs_property(self.sess, 'a')
    self.sess.get.assert_called_with(f'group_types/{GROUP_TYPE['id']}/group_specs/a', microversion=self.sess.default_microversion)
    self.assertEqual('b', rsp)