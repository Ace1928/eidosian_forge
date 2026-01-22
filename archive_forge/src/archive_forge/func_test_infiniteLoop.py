import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def test_infiniteLoop(self):
    """
        L{findMachinesViaWrapper} ignores infinite loops.

        Note this test can't fail - it can only run forever!
        """
    source = '\n        class InfiniteLoop(object):\n            pass\n\n        InfiniteLoop.loop = InfiniteLoop\n        '
    module = self.makeModule(source, self.pathDir, 'loop.py')
    self.assertFalse(list(self.findMachinesViaWrapper(module)))