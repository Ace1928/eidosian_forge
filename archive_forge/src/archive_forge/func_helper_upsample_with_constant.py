import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def helper_upsample_with_constant(self, raw_scale: bool=False) -> None:
    from_opset = 9
    to_opset = 8
    data_type = TensorProto.FLOAT
    scale_value = [1.0, 1.0, 2.0, 3.0]
    scale_tensor = onnx.helper.make_tensor('const_value', onnx.TensorProto.FLOAT, [4], bytes(struct.pack('4f', *scale_value)) if raw_scale else scale_value, raw_scale)
    nodes = [onnx.helper.make_node('Constant', inputs=[], outputs=['Constant_Output'], value=scale_tensor), onnx.helper.make_node('Upsample', inputs=['X', 'Constant_Output'], outputs=['Y'], mode='nearest')]
    graph = helper.make_graph(nodes, 'test_upsample', [onnx.helper.make_tensor_value_info('X', data_type, [1, 1, 2, 2])], [onnx.helper.make_tensor_value_info('Y', data_type, [1, 1, 4, 6])], value_info=[onnx.helper.make_tensor_value_info('Constant_Output', data_type, [4])])
    converted_model = self._converted(graph, helper.make_operatorsetid('', from_opset), to_opset)
    assert len(converted_model.graph.node) == 1
    assert converted_model.graph.node[0].op_type == 'Upsample'
    assert len(converted_model.graph.node[0].attribute) == 2
    assert converted_model.graph.node[0].attribute[1].name == 'scales'
    assert converted_model.opset_import[0].version == to_opset