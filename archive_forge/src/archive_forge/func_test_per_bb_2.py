import unittest
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.tests import TestCase
from . import refprune_proto as proto
def test_per_bb_2(self):
    mod, stats = self.check(self.per_bb_ir_2)
    self.assertEqual(stats.basicblock, 4)
    self.assertIn('call void @NRT_incref(i8* %ptr)', str(mod))