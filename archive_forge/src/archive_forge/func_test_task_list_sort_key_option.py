from osc_lib.cli import format_columns
from openstackclient.image.v2 import task
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
def test_task_list_sort_key_option(self):
    arglist = ['--sort-key', 'created_at']
    verifylist = [('sort_key', 'created_at')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.image_client.tasks.assert_called_with(sort_key=parsed_args.sort_key)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.datalist, data)