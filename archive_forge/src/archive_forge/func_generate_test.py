import unittest
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.tests import TestCase
from . import refprune_proto as proto
def generate_test(self, case_gen):
    nodes, edges, expected = case_gen()
    irmod = self.generate_ir(nodes, edges)
    outmod = self.apply_refprune(irmod)
    self.check(outmod, expected, nodes)