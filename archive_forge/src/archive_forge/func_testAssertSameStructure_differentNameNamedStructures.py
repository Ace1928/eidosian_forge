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
def testAssertSameStructure_differentNameNamedStructures(self):
    self.assertRaises(TypeError, tree.assert_same_structure, NestTest.SameNameab(0, 1), NestTest.NotSameName(2, 3))