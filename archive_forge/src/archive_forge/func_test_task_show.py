from osc_lib.cli import format_columns
from openstackclient.image.v2 import task
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
def test_task_show(self):
    arglist = [self.task.id]
    verifylist = [('task', self.task.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.image_client.get_task.assert_called_with(self.task.id)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)