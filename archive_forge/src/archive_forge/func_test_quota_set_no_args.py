import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import quota
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('octaviaclient.osc.v2.utils.get_quota_attrs')
def test_quota_set_no_args(self, mock_attrs):
    project_id = ['fake_project_id']
    mock_attrs.return_value = {'project_id': project_id}
    arglist = [project_id]
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertNotCalled(self.api_mock.quota_set)