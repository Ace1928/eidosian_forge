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
def testUnflattenSequenceAs_notIterableError(self):
    with self.assertRaisesRegex(TypeError, 'flat_sequence must be a sequence'):
        tree.unflatten_as('hi', 'bye')