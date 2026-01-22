from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import group_type
from openstack.tests.unit import base
def test_fetch_group_specs(self):
    sot = group_type.GroupType(**GROUP_TYPE)
    resp = mock.Mock()
    resp.body = {'group_specs': {'a': 'b', 'c': 'd'}}
    resp.json = mock.Mock(return_value=resp.body)
    resp.status_code = 200
    self.sess.get = mock.Mock(return_value=resp)
    rsp = sot.fetch_group_specs(self.sess)
    self.sess.get.assert_called_with(f'group_types/{GROUP_TYPE['id']}/group_specs', microversion=self.sess.default_microversion)
    self.assertEqual(resp.body['group_specs'], rsp.group_specs)
    self.assertIsInstance(rsp, group_type.GroupType)