import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def check_index_type(tp):
    index = ir.Constant(dbl, 1.0)
    with self.assertRaises(TypeError):
        tp.gep(index)