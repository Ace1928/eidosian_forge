import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_sum_8_5(self) -> None:
    nodes = [helper.make_node('Sum', ['data_0', 'data_1', 'data_2', 'data_3', 'data_4'], ['sum'])]
    graph = helper.make_graph(nodes, 'test', [helper.make_tensor_value_info('data_0', TensorProto.FLOAT, (5,)), helper.make_tensor_value_info('data_1', TensorProto.FLOAT, (5,)), helper.make_tensor_value_info('data_2', TensorProto.FLOAT, (5,)), helper.make_tensor_value_info('data_3', TensorProto.FLOAT, (5,)), helper.make_tensor_value_info('data_4', TensorProto.FLOAT, (5,))], [helper.make_tensor_value_info('sum', TensorProto.FLOAT, (5,))])
    converted_model = self._converted(graph, helper.make_operatorsetid('', 8), 5)
    assert converted_model.graph.node[0].op_type == 'Sum'
    assert converted_model.opset_import[0].version == 5