from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_set_boot_device(self):
    self.node.set_boot_device(self.session, 'pxe', persistent=False)
    self.session.put.assert_called_once_with('nodes/%s/management/boot_device' % self.node.id, json={'boot_device': 'pxe', 'persistent': False}, headers=mock.ANY, microversion=mock.ANY, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)