import sys
from io import BytesIO
from typing import List, Optional
from twisted.python import _shellcomp, reflect, usage
from twisted.python.usage import CompleteFiles, CompleteList, Completer, Completions
from twisted.trial import unittest
def test_brokenActions(self):
    """
        A C{Completer} with repeat=True may only be used as the
        last item in the extraActions list.
        """

    class BrokenActions(usage.Options):
        compData = usage.Completions(extraActions=[usage.Completer(repeat=True), usage.Completer()])
    outputFile = BytesIO()
    opts = BrokenActions()
    self.patch(opts, '_shellCompFile', outputFile)
    self.assertRaises(ValueError, opts.parseOptions, ['', '--_shell-completion', 'zsh:2'])