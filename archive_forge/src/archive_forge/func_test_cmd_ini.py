import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
def test_cmd_ini(self):
    self.complete(['brz', 'ini'])
    self.assertCompletionContains('init', 'init-shared-repo', 'init-shared-repository')
    self.assertCompletionOmits('commit')