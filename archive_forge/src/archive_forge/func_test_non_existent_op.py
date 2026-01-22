import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_non_existent_op(self) -> None:

    def test() -> None:
        nodes = [helper.make_node('Cos', ['X'], ['Y'])]
        graph = helper.make_graph(nodes, 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, (5,))], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, (5,))])
        self._converted(graph, helper.make_operatorsetid('', 8), 6)
    self.assertRaises(RuntimeError, test)