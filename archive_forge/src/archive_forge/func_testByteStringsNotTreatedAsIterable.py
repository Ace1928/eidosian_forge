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
def testByteStringsNotTreatedAsIterable(self):
    structure = [u'unicode string', b'byte string']
    flattened_structure = tree.flatten_up_to(structure, structure)
    self.assertEqual(structure, flattened_structure)