import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_axes_attr_to_input_12_13(self) -> None:
    nodes = [helper.make_node('ReduceSum', ['X'], ['Y'], axes=[0])]
    graph = helper.make_graph(nodes, 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, (5, 5))], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, (1, 5))])
    converted_model = self._converted(graph, helper.make_operatorsetid('', 12), 13)
    assert converted_model.graph.node[0].op_type == 'Constant'
    assert converted_model.opset_import[0].version == 13