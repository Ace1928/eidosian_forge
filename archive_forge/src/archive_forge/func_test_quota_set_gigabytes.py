from unittest import mock
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from osc_lib import exceptions
from manilaclient import api_versions
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.osc.v2 import quotas as osc_quotas
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_quota_set_gigabytes(self):
    arglist = [self.project.id, '--gigabytes', '1100']
    verifylist = [('project', self.project.id), ('gigabytes', 1100)]
    with mock.patch('osc_lib.utils.find_resource') as mock_find_resource:
        mock_find_resource.return_value = self.project
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.quotas_mock.update.assert_called_with(force=None, gigabytes=1100, share_networks=None, shares=None, snapshot_gigabytes=None, snapshots=None, tenant_id=self.project.id, user_id=None)
        self.assertIsNone(result)