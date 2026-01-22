from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
def test_sortsAttrs(self):
    """
        L{elementMaker} orders HTML attributes lexicographically.
        """
    expected = '<div a="1" b="2" c="3"></div>'
    self.assertEqual(expected, self.elementMaker('div', b='2', a='1', c='3'))