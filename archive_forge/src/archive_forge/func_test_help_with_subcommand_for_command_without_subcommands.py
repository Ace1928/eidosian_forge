from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.command import Command
import gslib.tests.testcase as testcase
def test_help_with_subcommand_for_command_without_subcommands(self):
    stdout = self.RunCommand('help', ['ls', 'asdf'], return_stdout=True)
    self.assertIn('has no subcommands', stdout)