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
def testAssertSameStructure_differentNumElements(self):
    with self.assertRaisesRegex(ValueError, 'The two structures don\'t have the same nested structure\\.\n\nFirst structure:.*?\n\nSecond structure:.*\n\nMore specifically: Substructure "type=tuple str=\\(\\(1, 2\\), 3\\)" is a sequence, while substructure "type=str str=spam" is not\nEntire first structure:\n\\(\\(\\(\\., \\.\\), \\.\\), \\., \\(\\., \\.\\)\\)\\nEntire second structure:\n\\(\\., \\.\\)'):
        tree.assert_same_structure(STRUCTURE1, STRUCTURE_DIFFERENT_NUM_ELEMENTS)