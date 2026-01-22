from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.command import Command
import gslib.tests.testcase as testcase
def test_help_subcommand_arg(self):
    stdout = self.RunCommand('help', ['web', 'set'], return_stdout=True)
    self.assertIn('gsutil web set', stdout)
    self.assertNotIn('gsutil web get', stdout)