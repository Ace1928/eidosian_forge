import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
def test_status_negated(self):
    dc = DataCollector()
    cmd = dc.command('status')
    self.assertSubset(['--no-versioned', '--no-verbose'], [str(o) for o in cmd.options])