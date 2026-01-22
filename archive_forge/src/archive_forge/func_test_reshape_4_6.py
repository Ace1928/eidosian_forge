import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_reshape_4_6(self) -> None:
    nodes = [helper.make_node('Reshape', ['X'], ['Y'], shape=[5])]
    graph = helper.make_graph(nodes, 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, (5,))], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, (5,))])
    converted_model = self._converted(graph, helper.make_operatorsetid('', 4), 6)
    assert converted_model.graph.node[0].op_type == 'Constant'
    assert converted_model.graph.node[1].op_type == 'Reshape'
    assert converted_model.opset_import[0].version == 6