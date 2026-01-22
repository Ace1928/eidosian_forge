import datetime
from unittest import mock
from openstackclient.compute.v2 import usage as usage_cmds
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_usage_show_with_options(self):
    arglist = ['--project', self.project.id, '--start', '2016-11-11', '--end', '2016-12-20']
    verifylist = [('project', self.project.id), ('start', '2016-11-11'), ('end', '2016-12-20')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.get_usage.assert_called_with(project=self.project.id, start=datetime.datetime(2016, 11, 11, 0, 0), end=datetime.datetime(2016, 12, 20, 0, 0))
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)