import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.common import quota
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
def test_delete__compute(self):
    """Delete compute quotas only"""
    arglist = ['--compute', self.projects[0].id]
    verifylist = [('service', 'compute'), ('project', self.projects[0].id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.assertIsNone(result)
    self.projects_mock.get.assert_called_once_with(self.projects[0].id)
    self.compute_quotas_mock.delete.assert_called_once_with(self.projects[0].id)
    self.volume_quotas_mock.delete.assert_not_called()
    self.network_client.delete_quota.assert_not_called()