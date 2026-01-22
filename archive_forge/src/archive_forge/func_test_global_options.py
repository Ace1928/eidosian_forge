import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
def test_global_options(self):
    dc = DataCollector()
    dc.global_options()
    self.assertSubset(['--no-plugins', '--builtin'], dc.data.global_options)