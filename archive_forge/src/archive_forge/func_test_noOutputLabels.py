from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
def test_noOutputLabels(self):
    """
        L{tableMaker} does not add a colspan attribute to the input
        label's cell or a second row if there no output labels.
        """
    table = self.tableMaker('input label', (), port=self.port)
    self.assertEqual(len(table.children), 1)
    inputLabelRow, = table.children
    self.assertNotIn('colspan', inputLabelRow.attributes)