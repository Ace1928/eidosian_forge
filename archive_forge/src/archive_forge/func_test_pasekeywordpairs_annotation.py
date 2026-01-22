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
def test_pasekeywordpairs_annotation(self):

    def spam(eggs: str='foo, bar'):
        pass
    defaults = inspection.getfuncprops('spam', spam).argspec.defaults
    self.assertEqual(repr(defaults[0]), '"foo, bar"')