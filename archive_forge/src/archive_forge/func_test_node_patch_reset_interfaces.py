from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
@mock.patch.object(resource.Resource, '_prepare_request', autospec=True)
@mock.patch.object(resource.Resource, '_commit', autospec=True)
def test_node_patch_reset_interfaces(self, mock__commit, mock_prepreq, mock_patch):
    patch = {'path': 'test'}
    self.node.patch(self.session, patch=patch, retry_on_conflict=True, reset_interfaces=True)
    mock_prepreq.assert_called_once()
    prepreq_kwargs = mock_prepreq.call_args[1]
    self.assertEqual(prepreq_kwargs['params'], [('reset_interfaces', True)])
    mock__commit.assert_called_once()
    commit_args = mock__commit.call_args[0]
    commit_kwargs = mock__commit.call_args[1]
    self.assertIn('1.45', commit_args)
    self.assertEqual(commit_kwargs['retry_on_conflict'], True)
    mock_patch.assert_not_called()