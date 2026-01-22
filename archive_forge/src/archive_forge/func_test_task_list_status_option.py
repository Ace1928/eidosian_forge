from osc_lib.cli import format_columns
from openstackclient.image.v2 import task
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
def test_task_list_status_option(self):
    arglist = ['--status', self.tasks[0].status]
    verifylist = [('status', self.tasks[0].status)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.image_client.tasks.assert_called_with(status=self.tasks[0].status)