import sys
from io import BytesIO
from typing import List, Optional
from twisted.python import _shellcomp, reflect, usage
from twisted.python.usage import CompleteFiles, CompleteList, Completer, Completions
from twisted.trial import unittest
def test_incompleteCommandLine_case3(self):
    """
        Completion still happens even if a command-line is given
        that would normally throw UsageError.

        Break subcommand detection in a different way by providing
        an invalid subcommand name.
        """
    outputFile = BytesIO()
    self.patch(usage.Options, '_shellCompFile', outputFile)
    opts = FighterAceOptions()
    self.assertRaises(SystemExit, opts.parseOptions, ['--fokker', 'unknown-subcommand', '--list-server', '--_shell-completion', 'zsh:4'])
    outputFile.seek(0)
    self.assertEqual(1, len(outputFile.read(1)))