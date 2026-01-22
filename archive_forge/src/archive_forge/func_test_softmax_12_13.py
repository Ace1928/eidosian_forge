import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_softmax_12_13(self) -> None:
    axis = 0
    nodes = [helper.make_node('Softmax', ['X'], ['Y'], axis=axis)]
    graph = helper.make_graph(nodes, 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, (1, 2, 3))], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, (1, 2, 3))])
    converted_model = self._converted(graph, helper.make_operatorsetid('', 11), 13)
    assert converted_model.graph.node[0].op_type == 'Shape'
    assert converted_model.graph.node[1].op_type == 'Flatten'
    assert converted_model.graph.node[1].attribute[0].name == 'axis'
    assert converted_model.graph.node[1].attribute[0].i == axis
    assert converted_model.graph.node[2].op_type == 'Softmax'
    assert converted_model.graph.node[2].attribute[0].name == 'axis'
    assert converted_model.graph.node[2].attribute[0].i == -1
    assert converted_model.graph.node[3].op_type == 'Reshape'
    assert converted_model.opset_import[0].version == 13