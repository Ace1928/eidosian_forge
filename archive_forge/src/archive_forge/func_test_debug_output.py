import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
def test_debug_output(self):
    data = CompletionData()
    self.assertEqual('', BashCodeGen(data, debug=False).debug_output())
    self.assertTrue(BashCodeGen(data, debug=True).debug_output())