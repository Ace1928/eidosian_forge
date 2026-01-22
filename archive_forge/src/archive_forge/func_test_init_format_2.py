import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
def test_init_format_2(self):
    self.complete(['brz', 'init', '--format', '=', '2', 'directory'], cword=4)
    self.assertCompletionContains('2a')
    self.assertCompletionOmits('1.9')