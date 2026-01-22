import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_greater_9_8(self) -> None:
    from_opset = 9
    to_opset = 8
    data_type = TensorProto.UINT64
    nodes = [onnx.helper.make_node('Greater', inputs=['X1', 'X2'], outputs=['Y'])]
    input_shape = [2, 3, 4]
    graph = helper.make_graph(nodes, 'test_greater', [onnx.helper.make_tensor_value_info('X1', data_type, input_shape), onnx.helper.make_tensor_value_info('X2', data_type, input_shape)], [onnx.helper.make_tensor_value_info('Y', TensorProto.BOOL, input_shape)])
    converted_model = self._converted(graph, helper.make_operatorsetid('', from_opset), to_opset)
    assert converted_model.graph.node[2].op_type == 'Greater'
    assert converted_model.graph.output[0].type.tensor_type.elem_type == TensorProto.BOOL
    assert converted_model.opset_import[0].version == to_opset