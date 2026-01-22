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
def test_get_encoding_ascii(self):
    self.assertEqual(inspection.get_encoding(encoding_ascii), 'ascii')
    self.assertEqual(inspection.get_encoding(encoding_ascii.foo), 'ascii')