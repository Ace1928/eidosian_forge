import unittest
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.tests import TestCase
from . import refprune_proto as proto
def test_per_bb_1(self):
    mod, stats = self.check(self.per_bb_ir_1)
    self.assertEqual(stats.basicblock, 2)