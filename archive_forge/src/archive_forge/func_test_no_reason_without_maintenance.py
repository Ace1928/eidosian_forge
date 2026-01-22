from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_no_reason_without_maintenance(self):
    self.node.maintenance_reason = 'Can I?'
    self.assertRaises(ValueError, self.node.commit, self.session)
    self.assertFalse(self.session.put.called)
    self.assertFalse(self.session.patch.called)