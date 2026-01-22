from unittest import mock
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from osc_lib import exceptions
from manilaclient import api_versions
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.osc.v2 import quotas as osc_quotas
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_quota_set_default_class_shares(self):
    arglist = ['default', '--class', '--shares', '40']
    verifylist = [('project', 'default'), ('quota_class', True), ('shares', 40)]
    with mock.patch('osc_lib.utils.find_resource') as mock_find_resource:
        mock_find_resource.return_value = self.project
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.quota_classes_mock.update.assert_called_with(class_name='default', gigabytes=None, share_networks=None, shares=40, snapshot_gigabytes=None, snapshots=None)
        self.assertIsNone(result)
        mock_find_resource.assert_not_called()
        self.quotas_mock.assert_not_called()