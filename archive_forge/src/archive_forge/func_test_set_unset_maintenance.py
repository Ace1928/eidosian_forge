from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_set_unset_maintenance(self):
    self.node.is_maintenance = True
    self.node.maintenance_reason = 'No work on Monday'
    self.node.commit(self.session)
    self.session.put.assert_called_once_with('nodes/%s/maintenance' % self.node.id, json={'reason': 'No work on Monday'}, headers=mock.ANY, microversion=mock.ANY)
    self.node.is_maintenance = False
    self.node.commit(self.session)
    self.assertIsNone(self.node.maintenance_reason)
    self.session.delete.assert_called_once_with('nodes/%s/maintenance' % self.node.id, json=None, headers=mock.ANY, microversion=mock.ANY)