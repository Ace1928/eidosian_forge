import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
def test_commands_from_plugins(self):
    dc = DataCollector()
    dc.commands()
    self.assertSubset(['bash-completion'], dc.data.all_command_aliases())