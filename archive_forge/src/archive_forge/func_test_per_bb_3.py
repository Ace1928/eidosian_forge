import unittest
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.tests import TestCase
from . import refprune_proto as proto
def test_per_bb_3(self):
    mod, stats = self.check(self.per_bb_ir_3)
    self.assertEqual(stats.basicblock, 2)
    self.assertIn('call void @NRT_decref(i8* %other)', str(mod))