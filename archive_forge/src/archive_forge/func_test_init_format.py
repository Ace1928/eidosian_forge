import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
def test_init_format(self):
    dc = DataCollector()
    cmd = dc.command('init')
    for opt in cmd.options:
        if opt.name == '--format':
            self.assertSubset(['2a'], opt.registry_keys)
            return
    raise AssertionError('Option --format not found')