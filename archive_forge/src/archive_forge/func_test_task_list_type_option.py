from osc_lib.cli import format_columns
from openstackclient.image.v2 import task
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
def test_task_list_type_option(self):
    arglist = ['--type', self.tasks[0].type]
    verifylist = [('type', self.tasks[0].type)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.image_client.tasks.assert_called_with(type=self.tasks[0].type)