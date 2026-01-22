from unittest import mock
from unittest.mock import call
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_group_delete_with_domain(self):
    get_mock_result = [exceptions.CommandError, self.groups[0]]
    self.groups_mock.get = mock.Mock(side_effect=get_mock_result)
    arglist = ['--domain', self.domain.id, self.groups[0].id]
    verifylist = [('domain', self.groups[0].domain_id), ('groups', [self.groups[0].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.groups_mock.get.assert_any_call(self.groups[0].id, domain_id=self.domain.id)
    self.groups_mock.delete.assert_called_once_with(self.groups[0].id)
    self.assertIsNone(result)