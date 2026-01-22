import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_batch_normalization_8_9(self) -> None:
    from_opset = 8
    to_opset = 9
    data_type = TensorProto.FLOAT
    nodes = [helper.make_node('BatchNormalization', inputs=['x', 's', 'bias', 'mean', 'var'], outputs=['y'])]
    input_shape = (1, 2, 1, 3)
    x = helper.make_tensor_value_info('x', data_type, input_shape)
    scale = helper.make_tensor_value_info('s', data_type, [input_shape[1]])
    B = helper.make_tensor_value_info('bias', data_type, [input_shape[1]])
    mean = helper.make_tensor_value_info('mean', data_type, [input_shape[1]])
    var = helper.make_tensor_value_info('var', data_type, [input_shape[1]])
    y = helper.make_tensor_value_info('y', data_type, input_shape)
    graph = helper.make_graph(nodes, 'test_batchnormalization_8_9', [x, scale, B, mean, var], [y])
    converted_model = self._converted(graph, helper.make_operatorsetid('', from_opset), to_opset)
    assert converted_model.graph.node[0].op_type == 'BatchNormalization'
    assert converted_model.opset_import[0].version == to_opset