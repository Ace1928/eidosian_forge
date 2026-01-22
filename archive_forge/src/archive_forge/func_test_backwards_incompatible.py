import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_backwards_incompatible(self) -> None:

    def test() -> None:
        nodes = [helper.make_node('Add', ['W', 'Z'], ['shape']), helper.make_node('Reshape', ['X', 'shape'], ['A']), helper.make_node('Add', ['A', 'W'], ['Y'])]
        graph = helper.make_graph(nodes, 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, (5,)), helper.make_tensor_value_info('W', TensorProto.FLOAT, (1,)), helper.make_tensor_value_info('Z', TensorProto.FLOAT, (1,))], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, (5,))])
        self._converted(graph, helper.make_operatorsetid('', 8), 2)
    self.assertRaises(RuntimeError, test)