import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_slice_9_10(self) -> None:
    nodes = [helper.make_node('Slice', ['X'], ['Y'], axes=[0, 1], starts=[0, 0], ends=[3, 10])]
    graph = helper.make_graph(nodes, 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, (20, 10, 5))], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, (3, 10, 5))])
    converted_model = self._converted(graph, helper.make_operatorsetid('', 9), 10)
    assert converted_model.graph.node[0].op_type == 'Constant'
    assert converted_model.graph.node[1].op_type == 'Constant'
    assert converted_model.graph.node[2].op_type == 'Constant'
    assert converted_model.graph.node[3].op_type == 'Slice'
    assert converted_model.opset_import[0].version == 10
    assert len(converted_model.graph.node[3].input) == 4
    assert len(converted_model.graph.node[3].attribute) == 0