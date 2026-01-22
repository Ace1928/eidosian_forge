from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_node_set_secure_boot_invalid_none(self):
    self.assertRaises(ValueError, self.node.set_secure_boot, self.session, None)