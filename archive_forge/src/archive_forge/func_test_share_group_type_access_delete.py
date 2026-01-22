from osc_lib import exceptions
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.osc.v2 import share_group_type_access as osc_sgta
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_group_type_access_delete(self):
    arglist = [self.share_group_type.id, self.project.id]
    verifylist = [('share_group_type', self.share_group_type.id), ('projects', [self.project.id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.type_access_mock.remove_project_access.assert_called_with(self.share_group_type, self.project.id)
    self.assertIsNone(result)