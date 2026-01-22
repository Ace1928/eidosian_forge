from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.command import Command
import gslib.tests.testcase as testcase
def test_help_runs_for_all_commands(self):
    for command in Command.__subclasses__():
        self.RunGsUtil(['help', command.command_spec.command_name])