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
def testAssertSameStructure_listStructureWithAndWithoutTypes(self):
    structure1_list = [[[1, 2], 3], 4, [5, 6]]
    with self.assertRaisesRegex(TypeError, "don't have the same sequence type"):
        tree.assert_same_structure(STRUCTURE1, structure1_list)
    tree.assert_same_structure(STRUCTURE1, STRUCTURE2, check_types=False)
    tree.assert_same_structure(STRUCTURE1, structure1_list, check_types=False)