import collections
import doctest
import types
from typing import Any, Iterator, Mapping
import unittest
from absl.testing import parameterized
import attr
import numpy as np
import tree
import wrapt
def testAssertSameStructure_listVsNdArray(self):
    with self.assertRaisesRegex(ValueError, 'The two structures don\'t have the same nested structure\\.\n\nFirst structure:.*?\n\nSecond structure:.*\n\nMore specifically: Substructure "type=list str=\\[0, 1\\]" is a sequence, while substructure "type=ndarray str=\\[0 1\\]" is not'):
        tree.assert_same_structure([0, 1], np.array([0, 1]))