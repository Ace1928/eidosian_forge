import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_add_5_8_with_unsqueeze(self) -> None:
    nodes = [helper.make_node('Add', ['X1', 'X2'], ['Y'], axis=0, broadcast=1)]
    graph = helper.make_graph(nodes, 'test', [helper.make_tensor_value_info('X1', TensorProto.FLOAT, (5, 2)), helper.make_tensor_value_info('X2', TensorProto.FLOAT, (5,))], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, (5,))])
    converted_model = self._converted(graph, helper.make_operatorsetid('', 5), 8)
    assert converted_model.graph.node[0].op_type == 'Unsqueeze'
    assert converted_model.graph.node[1].op_type == 'Add'
    assert converted_model.opset_import[0].version == 8