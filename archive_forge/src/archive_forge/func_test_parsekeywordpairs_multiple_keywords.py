import inspect
import os
import sys
import unittest
from collections.abc import Sequence
from typing import List
from bpython import inspection
from bpython.test.fodder import encoding_ascii
from bpython.test.fodder import encoding_latin1
from bpython.test.fodder import encoding_utf8
def test_parsekeywordpairs_multiple_keywords(self):

    def spam(eggs=23, foobar='yay'):
        pass
    defaults = inspection.getfuncprops('spam', spam).argspec.defaults
    self.assertEqual(repr(defaults[0]), '23')
    self.assertEqual(repr(defaults[1]), '"yay"')