import datetime
from unittest import mock
from openstackclient.compute.v2 import usage as usage_cmds
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_usage_list_no_options(self):
    arglist = []
    verifylist = [('start', None), ('end', None)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.projects_mock.list.assert_called_with()
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(tuple(self.data), tuple(data))