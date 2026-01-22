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
@unittest.skipUnless(numpy is not None and numpy.__version__ >= '1.18', 'requires numpy >= 1.18')
def test_getfuncprops_numpy_array(self):
    props = inspection.getfuncprops('array', numpy.array)
    self.assertEqual(props.func, 'array')
    self.assertEqual(props.argspec.args, ['object', 'dtype'])