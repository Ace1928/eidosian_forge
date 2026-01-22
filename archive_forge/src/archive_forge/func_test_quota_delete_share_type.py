from unittest import mock
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from osc_lib import exceptions
from manilaclient import api_versions
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.osc.v2 import quotas as osc_quotas
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_quota_delete_share_type(self):
    arglist = [self.project.id, '--share-type', 'default']
    verifylist = [('project', self.project.id), ('share_type', 'default')]
    with mock.patch('osc_lib.utils.find_resource') as mock_find_resource:
        mock_find_resource.return_value = self.project
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.quotas_mock.delete.assert_called_with(share_type='default', tenant_id=self.project.id, user_id=None)
        self.assertIsNone(result)