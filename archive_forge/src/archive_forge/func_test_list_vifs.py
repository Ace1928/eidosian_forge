from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_list_vifs(self):
    self.session.get.return_value.json.return_value = {'vifs': [{'id': '1234'}, {'id': '5678'}]}
    res = self.node.list_vifs(self.session)
    self.assertEqual(['1234', '5678'], res)
    self.session.get.assert_called_once_with('nodes/%s/vifs' % self.node.id, headers=mock.ANY, microversion='1.28')