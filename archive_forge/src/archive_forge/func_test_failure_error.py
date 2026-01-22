from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_failure_error(self, mock_fetch):

    def _get_side_effect(_self, session):
        self.node.provision_state = 'error'
        self.assertIs(session, self.session)
    mock_fetch.side_effect = _get_side_effect
    self.assertRaisesRegex(exceptions.ResourceFailure, 'failure state "error"', self.node.wait_for_provision_state, self.session, 'manageable')