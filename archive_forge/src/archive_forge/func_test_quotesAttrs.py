from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
def test_quotesAttrs(self):
    """
        L{elementMaker} quotes HTML attributes according to DOT's quoting rule.

        See U{http://www.graphviz.org/doc/info/lang.html}, footnote 1.
        """
    expected = '<div a="1" b="a \\" quote" c="a string"></div>'
    self.assertEqual(expected, self.elementMaker('div', b='a " quote', a=1, c='a string'))