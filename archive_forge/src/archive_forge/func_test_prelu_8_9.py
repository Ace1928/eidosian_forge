import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_prelu_8_9(self) -> None:
    from_opset = 8
    to_opset = 9
    data_type = TensorProto.FLOAT
    nodes = [onnx.helper.make_node('PRelu', inputs=['X', 'Slope'], outputs=['Y'])]
    input_shape = [2, 3, 4]
    graph = helper.make_graph(nodes, 'test_prelu', [onnx.helper.make_tensor_value_info('X', data_type, input_shape), onnx.helper.make_tensor_value_info('Slope', data_type, input_shape)], [onnx.helper.make_tensor_value_info('Y', data_type, input_shape)])
    converted_model = self._converted(graph, helper.make_operatorsetid('', from_opset), to_opset)
    assert converted_model.graph.node[0].op_type == 'PRelu'
    assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
    assert converted_model.opset_import[0].version == to_opset