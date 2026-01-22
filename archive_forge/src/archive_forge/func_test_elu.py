import unittest
from onnx import checker, defs, helper
def test_elu(self) -> None:
    self.assertTrue(defs.has('Elu'))
    node_def = helper.make_node('Elu', ['X'], ['Y'], alpha=1.0)
    checker.check_node(node_def)