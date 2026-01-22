import datetime
from unittest import mock
from openstackclient.compute.v2 import usage as usage_cmds
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_usage_show_no_options(self):
    self.app.client_manager.auth_ref = mock.Mock()
    self.app.client_manager.auth_ref.project_id = self.project.id
    arglist = []
    verifylist = [('project', None), ('start', None), ('end', None)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)