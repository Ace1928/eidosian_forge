from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
def test_quietHidesOutput(self):
    """
        Passing -q/--quiet hides all output.
        """
    self.tool(argv=[self.fakeFQPN, '--quiet'])
    self.assertFalse(self.collectedOutput)
    self.tool(argv=[self.fakeFQPN, '-q'])
    self.assertFalse(self.collectedOutput)