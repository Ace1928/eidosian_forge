import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
def test_command_cases(self):
    data = CompletionData()
    bar = CommandData('bar')
    bar.aliases.append('baz')
    bar.options.append(OptionData('--opt'))
    data.commands.append(bar)
    data.commands.append(CommandData('foo'))
    cg = BashCodeGen(data)
    self.assertEqualDiff('\tbar|baz)\n\t\tcmdOpts=( --opt )\n\t\t;;\n\tfoo)\n\t\tcmdOpts=(  )\n\t\t;;\n', cg.command_cases())