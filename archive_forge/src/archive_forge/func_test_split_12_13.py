import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_split_12_13(self) -> None:
    nodes = [helper.make_node('Split', ['X'], ['Y1', 'Y2'], split=[2, 3])]
    graph = helper.make_graph(nodes, 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, (5,))], [helper.make_tensor_value_info('Y1', TensorProto.FLOAT, (2,)), helper.make_tensor_value_info('Y2', TensorProto.FLOAT, (3,))])
    converted_model = self._converted(graph, helper.make_operatorsetid('', 12), 13)
    assert converted_model.graph.node[0].op_type == 'Constant'
    assert converted_model.graph.node[1].op_type == 'Split'
    assert converted_model.opset_import[0].version == 13