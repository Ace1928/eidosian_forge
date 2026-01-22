from osc_lib.cli import format_columns
from openstackclient.image.v2 import task
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
def test_task_list_sort_dir_option(self):
    arglist = ['--sort-dir', 'desc']
    verifylist = [('sort_dir', 'desc')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.image_client.tasks.assert_called_with(sort_dir=parsed_args.sort_dir)