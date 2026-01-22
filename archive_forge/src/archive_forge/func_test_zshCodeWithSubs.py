import sys
from io import BytesIO
from typing import List, Optional
from twisted.python import _shellcomp, reflect, usage
from twisted.python.usage import CompleteFiles, CompleteList, Completer, Completions
from twisted.trial import unittest
def test_zshCodeWithSubs(self):
    """
        Generate a completion function with subcommands,
        and test the textual output against a known correct output
        """
    outputFile = BytesIO()
    self.patch(usage.Options, '_shellCompFile', outputFile)
    self.patch(sys, 'argv', ['silly2', '', '--_shell-completion', 'zsh:2'])
    opts = SimpleProgWithSubcommands()
    self.assertRaises(SystemExit, opts.parseOptions)
    self.assertEqual(testOutput2, outputFile.getvalue())