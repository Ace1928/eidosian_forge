import os.path
import unittest
from Cython.TestUtils import TransformTest
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.ParseTreeTransforms import _calculate_pickle_checksums
from Cython.Compiler.Nodes import *
from Cython.Compiler import Main, Symtab, Options
def test_calculate_pickle_checksums(self):
    checksums = _calculate_pickle_checksums(['member1', 'member2', 'member3'])
    assert 2 <= len(checksums) <= 3, checksums