from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_power_on(self):
    self.node.set_power_state(self.session, 'power on')
    self.session.put.assert_called_once_with('nodes/%s/states/power' % FAKE['uuid'], json={'target': 'power on'}, headers=mock.ANY, microversion=None, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)